#define GLM_FORCE_CUDA
#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <glm/glm.hpp>
#include "utilityCore.hpp"
#include "kernel.h"

#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

// LOOK-2.1 potentially useful for doing grid-based neighbor search
#ifndef imax
#define imax( a, b ) ( ((a) > (b)) ? (a) : (b) )
#endif

#ifndef imin
#define imin( a, b ) ( ((a) < (b)) ? (a) : (b) )
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)

/**
* Check for CUDA errors; print and exit if there was a problem.
*/
void checkCUDAError(const char *msg, int line = -1) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    if (line >= 0) {
      fprintf(stderr, "Line %d: ", line);
    }
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}


/*****************
* Configuration *
*****************/

/*! Block size used for CUDA kernel launch. */
#define blockSize 128

// LOOK-1.2 Parameters for the boids algorithm.
// These worked well in our reference implementation.
#define rule1Distance 5.0f
#define rule2Distance 3.0f
#define rule3Distance 5.0f

#define rule1Scale 0.01f
#define rule2Scale 0.1f
#define rule3Scale 0.1f

#define maxSpeed 1.0f

/*! Size of the starting area in simulation space. */
#define scene_scale 100.0f

/***********************************************
* Kernel state (pointers are device pointers) *
***********************************************/

int numObjects;
dim3 threadsPerBlock(blockSize);

// LOOK-1.2 - These buffers are here to hold all your boid information.
// These get allocated for you in Boids::initSimulation.
// Consider why you would need two velocity buffers in a simulation where each
// boid cares about its neighbors' velocities.
// These are called ping-pong buffers.
glm::vec3 *dev_pos;
glm::vec3 *dev_vel1;
glm::vec3 *dev_vel2;

// LOOK-2.1 - these are NOT allocated for you. You'll have to set up the thrust
// pointers on your own too.

// For efficient sorting and the uniform grid. These should always be parallel.
//Buffer containing a pointer for each boid to its data in dev_pos and dev_vel1 and dev_vel2
int *dev_particleArrayIndices; // What index in dev_pos and dev_velX represents this particle?
// Buffer containing the grid index for each boid.
int *dev_particleGridIndices; // What grid cell is this particle in?
// needed for use with thrust
thrust::device_ptr<int> dev_thrust_particleArrayIndices;
thrust::device_ptr<int> dev_thrust_particleGridIndices;

// Buffer containing a pointer for each cell to the begining of dev_particleArrayIndices
int *dev_gridCellStartIndices; // What part of dev_particleArrayIndices belongs
// Buffer containing a pointer for each cell to the end of dev_particleArrayIndices
int *dev_gridCellEndIndices;   // to this cell?

// TODO-2.3 - consider what additional buffers you might need to reshuffle
// the position and velocity data to be coherent within cells.

// LOOK-2.1 - Grid parameters based on simulation parameters.
// These are automatically computed for you in Boids::initSimulation
int gridCellCount;
int gridSideCount;
float gridCellWidth;
float gridInverseCellWidth;
glm::vec3 gridMinimum;

/******************
* initSimulation *
******************/

__host__ __device__ unsigned int hash(unsigned int a) {
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >> 16);
  return a;
}

/**
* LOOK-1.2 - this is a typical helper function for a CUDA kernel.
* Function for generating a random vec3.
*/
__host__ __device__ glm::vec3 generateRandomVec3(float time, int index) {
  thrust::default_random_engine rng(hash((int)(index * time)));
  thrust::uniform_real_distribution<float> unitDistrib(-1, 1);

  return glm::vec3((float)unitDistrib(rng), (float)unitDistrib(rng), (float)unitDistrib(rng));
}

/**
* LOOK-1.2 - This is a basic CUDA kernel.
* CUDA kernel for generating boids with a specified mass randomly around the star.
*/
__global__ void kernGenerateRandomPosArray(int time, int N, glm::vec3 * arr, float scale) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    glm::vec3 rand = generateRandomVec3(time, index);
    arr[index].x = scale * rand.x;
    arr[index].y = scale * rand.y;
    arr[index].z = scale * rand.z;
  }
}

/**
* Initialize memory, update some globals
*/
void Boids::initSimulation(int N) {
  numObjects = N;
  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  // LOOK-1.2 - This is basic CUDA memory management and error checking.
  // Don't forget to cudaFree in  Boids::endSimulation.
  cudaMalloc((void**)&dev_pos, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_pos failed!");

  cudaMalloc((void**)&dev_vel1, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel1 failed!");

  cudaMalloc((void**)&dev_vel2, N * sizeof(glm::vec3));
  checkCUDAErrorWithLine("cudaMalloc dev_vel2 failed!");

  // LOOK-1.2 - This is a typical CUDA kernel invocation.
  kernGenerateRandomPosArray<<<fullBlocksPerGrid, blockSize>>>(1, numObjects,
 dev_pos, scene_scale);
  checkCUDAErrorWithLine("kernGenerateRandomPosArray failed!");

  // LOOK-2.1 computing grid params
  gridCellWidth = 2.0f * std::max(std::max(rule1Distance, rule2Distance), rule3Distance);
  int halfSideCount = (int)(scene_scale / gridCellWidth) + 1;
  gridSideCount = 2 * halfSideCount;

  gridCellCount = gridSideCount * gridSideCount * gridSideCount;
  gridInverseCellWidth = 1.0f / gridCellWidth;
  float halfGridWidth = gridCellWidth * halfSideCount;
  gridMinimum.x -= halfGridWidth;
  gridMinimum.y -= halfGridWidth;
  gridMinimum.z -= halfGridWidth;

  // TODO-2.1 TODO-2.3 - Allocate additional buffers here.
  //Uniform Grid Buffers
  cudaMalloc((void**)&dev_particleArrayIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleArrayIndices failed!");

  cudaMalloc((void**)&dev_particleGridIndices, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_particleGridIndices failed!");

  cudaMalloc((void**)&dev_gridCellStartIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellStartIndices failed!");

  cudaMalloc((void**)&dev_gridCellEndIndices, gridCellCount * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_gridCellEndIndices failed!");

  // Thrust buffers, used for prallel sorting
  dev_thrust_particleArrayIndices = thrust::device_pointer_cast<int>(dev_particleArrayIndices);
  dev_thrust_particleGridIndices  = thrust::device_pointer_cast<int>(dev_particleGridIndices);

  cudaDeviceSynchronize();
}


/******************
* copyBoidsToVBO *
******************/

/**
* Copy the boid positions into the VBO so that they can be drawn by OpenGL.
*/
__global__ void kernCopyPositionsToVBO(int N, glm::vec3 *pos, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  float c_scale = -1.0f / s_scale;

  if (index < N) {
    vbo[4 * index + 0] = pos[index].x * c_scale;
    vbo[4 * index + 1] = pos[index].y * c_scale;
    vbo[4 * index + 2] = pos[index].z * c_scale;
    vbo[4 * index + 3] = 1.0f;
  }
}

__global__ void kernCopyVelocitiesToVBO(int N, glm::vec3 *vel, float *vbo, float s_scale) {
  int index = threadIdx.x + (blockIdx.x * blockDim.x);

  if (index < N) {
    vbo[4 * index + 0] = vel[index].x + 0.3f;
    vbo[4 * index + 1] = vel[index].y + 0.3f;
    vbo[4 * index + 2] = vel[index].z + 0.3f;
    vbo[4 * index + 3] = 1.0f;
  }
}

/**
* Wrapper for call to the kernCopyboidsToVBO CUDA kernel.
*/
void Boids::copyBoidsToVBO(float *vbodptr_positions, float *vbodptr_velocities) {
  dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

  kernCopyPositionsToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_pos, vbodptr_positions, scene_scale);
  kernCopyVelocitiesToVBO << <fullBlocksPerGrid, blockSize >> >(numObjects, dev_vel1, vbodptr_velocities, scene_scale);

  checkCUDAErrorWithLine("copyBoidsToVBO failed!");

  cudaDeviceSynchronize();
}


/******************
* stepSimulation *
******************/

/**
* boidCohesionRuleNaive()
* boids move towards the perceived center of mass of their neighbors
* 
* Cohesion depends soley on the position of each boid, so we want to calculate the center of mass.
* Assuming each boid weighs the same, the center of mass is simply the average position.
* Therefore, we add each component of each boid and divide.
* NOTE: For the Naive implementation, this means each thread is going to be doing the same exact work.
*       That is super bad and goes against the idea of distributing work. This becomes a non-issue
*       when each boid looks only at their local neighbors, as each boid will have a different subset
*       to look at.
*/
__device__ glm::vec3 boidCohesionRuleNaive(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	glm::vec3 result(0.0f);
	glm::vec3 perceived_center(0.0f);
	glm::vec3 selfPos = pos[iSelf];
	float neighbors = 0.0f;

	for (int i = 0; i < N; i++) {
		if ((i != iSelf) && (glm::distance(selfPos, pos[i]) < rule1Distance)) {
			perceived_center += pos[i];
			neighbors++;
		}
	}

	if (neighbors) {
		perceived_center /= neighbors;
		result = (perceived_center - selfPos) * rule1Scale;
	}

	return result;
}


/**
* boidCohesionRuleGrid()
* boids move towards the perceived center of mass of their neighbors
*
* Cohesion depends soley on the position of each boid, so we want to calculate the center of mass.
* Assuming each boid weighs the same, the center of mass is simply the average position.
* Therefore, we add each component of each boid and divide.
*/
__device__ glm::vec3 boidCohesionRuleGrid(int N, int iSelf, const int *boidIndices, int b_start, int b_end, glm::vec3 *pos, const glm::vec3 *vel) {
	glm::vec3 result(0.0f);
	glm::vec3 perceived_center(0.0f);
	glm::vec3 selfPos = pos[iSelf];
	float neighbors = 0.0f;

	for (int i = b_start; i < b_end; i++) {
		int boid_idx = boidIndices[i];
		if ((boid_idx != iSelf) && (glm::distance(selfPos, pos[boid_idx]) < rule1Distance)) {
			perceived_center += pos[boid_idx];
			neighbors++;
		}
	}

	if (neighbors) {
		perceived_center /= neighbors;
		result = (perceived_center - selfPos) * rule1Scale;
	}

	return result;
}

/**
* boidSeperationRuleNaive()
* boids avoid getting to close to their neighbors
*
* In this rule, the boid is repulsed by nearby boids. To represent that, we take the distance
* between the boid and the neighbor boids and add the disance between the two as a sacled negative velocity.
* This has the effect of pushing each boid away from his neighbors. Note that a boid on either side will contribute
* to opposite directions.
*/
__device__ glm::vec3 boidSeperationRuleNaive(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	glm::vec3 result(0.0f);
	glm::vec3 seperation(0.0f);
	glm::vec3 selfPos = pos[iSelf];
	float neighbors = 0.0f;

	for (int i = 0; i < N; i++) {
		if ((i != iSelf) && (glm::distance(selfPos, pos[i]) < rule2Distance)) {
			seperation -= pos[i] - selfPos;
			neighbors++;
		}
	}

	if (neighbors) {
		result = seperation * rule2Scale;
	}

	return result;
}

/**
* boidSeperationRuleGrid()
* boids avoid getting to close to their neighbors
*
* In this rule, the boid is repulsed by nearby boids. To represent that, we take the distance
* between the boid and the neighbor boids and add the disance between the two as a sacled negative velocity.
* This has the effect of pushing each boid away from his neighbors. Note that a boid on either side will contribute
* to opposite directions.
*/
__device__ glm::vec3 boidSeperationRuleGrid(int N, int iSelf, const int *boidIndices, int b_start, int b_end, glm::vec3 *pos, const glm::vec3 *vel) {
	glm::vec3 result(0.0f);
	glm::vec3 seperation(0.0f);
	glm::vec3 selfPos = pos[iSelf];
	float neighbors = 0.0f;

	for (int i = b_start; i < b_end; i++) {
		int boid_idx = boidIndices[i];
		if ((boid_idx != iSelf) && (glm::distance(selfPos, pos[boid_idx]) < rule2Distance)) {
			seperation -= pos[boid_idx] - selfPos;
			neighbors++;
		}
	}

	if (neighbors) {
		result = seperation * rule2Scale;
	}

	return result;
}


/**
* boidAlignmentRuleNaive()
* boids generally try to move with the same direction and speed as their neighbors
*
* Boids want to match the velocit of their neighbors at t=a, so they will adjust their velocity accordingly.
* After each round, at t=a+dt, each boid will apply their change.
*/
__device__ glm::vec3 boidAlignmentRuleNaive(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
	glm::vec3 result(0.0f);
	glm::vec3 perceived_velocity(0.0f);
	glm::vec3 selfPos = pos[iSelf];
	glm::vec3 selfVelocity = vel[iSelf];
	float neighbors = 0.0f;

	for (int i = 0; i < N; i++) {
		if ((i != iSelf) && (glm::distance(selfPos, pos[i]) < rule3Distance)) {
			perceived_velocity += vel[i];
			neighbors++;
		}
	}

	if (neighbors) {
		perceived_velocity /= neighbors;
		result = perceived_velocity * rule3Scale;
	}

	return result;
}

/**
* boidAlignmentRuleGrid()
* boids generally try to move with the same direction and speed as their neighbors
*
* Boids want to match the velocit of their neighbors at t=a, so they will adjust their velocity accordingly.
* After each round, at t=a+dt, each boid will apply their change.
*/
__device__ glm::vec3 boidAlignmentRuleGrid(int N, int iSelf, const int *boidIndices, int b_start, int b_end, glm::vec3 *pos, const glm::vec3 *vel) {
	glm::vec3 result(0.0f);
	glm::vec3 perceived_velocity(0.0f);
	glm::vec3 selfPos = pos[iSelf];
	glm::vec3 selfVelocity = vel[iSelf];
	float neighbors = 0.0f;

	for (int i = b_start; i < b_end; i++) {
		int boid_idx = boidIndices[i];
		if ((boid_idx != iSelf) && (glm::distance(selfPos, pos[boid_idx]) < rule3Distance)) {
			perceived_velocity += vel[boid_idx];
			neighbors++;
		}
	}

	if (neighbors) {
		perceived_velocity /= neighbors;
		result = perceived_velocity * rule3Scale;
	}

	return result;
}


/**
* LOOK-1.2 You can use this as a helper for kernUpdateVelocityBruteForce.
* __device__ code can be called from a __global__ context
* Compute the new velocity on the body with index `iSelf` due to the `N` boids
* in the `pos` and `vel` arrays.
*/
__device__ glm::vec3 computeVelocityChange(int N, int iSelf, const glm::vec3 *pos, const glm::vec3 *vel) {
  // Rule 1: boids fly towards their local perceived center of mass, which excludes themselves
  // Rule 2: boids try to stay a distance d away from each other
  // Rule 3: boids try to match the speed of surrounding boids

	glm::vec3 delta(0.0f);

	// Apply each rule.
	delta += boidCohesionRuleNaive(N, iSelf, pos, vel);
	delta += boidSeperationRuleNaive(N, iSelf, pos, vel);
	delta += boidAlignmentRuleNaive(N, iSelf, pos, vel);

	return delta;
}

/**
* TODO-1.2 implement basic flocking
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdateVelocityBruteForce(int N, glm::vec3 *pos,
  glm::vec3 *vel1, glm::vec3 *vel2) {
  // Compute a new velocity based on pos and vel1
  // Clamp the speed
  // Record the new velocity into vel2. Question: why NOT vel1? 
  // Answer: Other threads may still be reading vel1!!

	int index = threadIdx.x + (blockIdx.x * blockDim.x);
	if (index >= N) {
		return;
	}

	glm::vec3 curntV = vel1[index];
	glm::vec3 deltaV = computeVelocityChange(N, index, pos, vel1);
	glm::vec3 newV = curntV + deltaV;

	// Clamp the speed. We do it this way to ensure that the total velocity is clamped,
	// not just the velocity in each direction (otherwise glm::clamp would be nice).
	if (glm::length(newV) > maxSpeed) {
		newV = glm::normalize(newV) * maxSpeed;
	}
	
	vel2[index] = newV;
}

/**
* LOOK-1.2 Since this is pretty trivial, we implemented it for you.
* For each of the `N` bodies, update its position based on its current velocity.
*/
__global__ void kernUpdatePos(int N, float dt, glm::vec3 *pos, glm::vec3 *vel) {
  // Update position by velocity
  int index = threadIdx.x + (blockIdx.x * blockDim.x);
  if (index >= N) {
    return;
  }
  glm::vec3 thisPos = pos[index];
  thisPos += vel[index] * dt;

  // Wrap the boids around so we don't lose them
  thisPos.x = thisPos.x < -scene_scale ? scene_scale : thisPos.x;
  thisPos.y = thisPos.y < -scene_scale ? scene_scale : thisPos.y;
  thisPos.z = thisPos.z < -scene_scale ? scene_scale : thisPos.z;

  thisPos.x = thisPos.x > scene_scale ? -scene_scale : thisPos.x;
  thisPos.y = thisPos.y > scene_scale ? -scene_scale : thisPos.y;
  thisPos.z = thisPos.z > scene_scale ? -scene_scale : thisPos.z;

  pos[index] = thisPos;
}

// LOOK-2.1 Consider this method of computing a 1D index from a 3D grid index.
// LOOK-2.3 Looking at this method, what would be the most memory efficient
//          order for iterating over neighboring grid cells?
//          for(x)
//            for(y)
//             for(z)? Or some other order?
__device__ int gridIndex3Dto1D(int x, int y, int z, int gridResolution) {
  return x + y * gridResolution + z * gridResolution * gridResolution;
}

__global__ void kernComputeIndices(int N, int gridResolution,
  glm::vec3 gridMin, float inverseCellWidth,
  glm::vec3 *pos, int *indices, int *gridIndices) {
    // TODO-2.1
    // - Label each boid with the index of its grid cell.
    // - Set up a parallel array of integer indices as pointers to the actual
    //   boid data in pos and vel1/vel2
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		glm::vec3 myGridPos = pos[index];
		int myGridIdx = 0;

		// Add grid minimum to move +/- halfWidth range to 0 - fullWidth range.
		myGridPos -= gridMin;
		// Cells are cubes, so all dimensions are identical, divide each pos by cell width
		myGridPos *= inverseCellWidth;
		// Round down to throw away float garbage!
		myGridPos = glm::floor(myGridPos);

		// Compute a 1D index from the 3D index
		myGridIdx = gridIndex3Dto1D(myGridPos.x, myGridPos.y, myGridPos.z, gridResolution);

		// Store the grid index in the indices buffer using the boid IDX as the key
		// and the index in and index buffer. These two will be sorted in parallel.
		// The end result will be that indices will be sorted by myGridIdx so consecutive
		// boids will have their indices colocated in memory.
		gridIndices[index] = myGridIdx;
		indices[index] = index;
	}
}

// LOOK-2.1 Consider how this could be useful for indicating that a cell
//          does not enclose any boids
__global__ void kernResetIntBuffer(int N, int *intBuffer, int value) {
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if (index < N) {
    intBuffer[index] = value;
  }
}

__global__ void kernIdentifyCellStartEnd(int N, int *particleGridIndices,
  int *gridCellStartIndices, int *gridCellEndIndices) {
  // TODO-2.1
  // Identify the start point of each cell in the gridIndices array.
  // This is basically a parallel unrolling of a loop that goes
  // "this index doesn't match the one before it, must be a new cell!"
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		// Collect some information
		int myGridIdx = particleGridIndices[index];

		// Always Start if first cell
		if ((index == 0) || particleGridIndices[index] != particleGridIndices[index - 1]) {
			gridCellStartIndices[myGridIdx] = index; // Start of grid myGridIdx is boid at index
		}

		// Always End if last cell
		if ((index == (N - 1)) || (particleGridIndices[index] != particleGridIndices[index + 1])) {
			gridCellEndIndices[myGridIdx] = index;
		}
	}
}

__global__ void kernUpdateVelNeighborSearchScattered(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  int *particleArrayIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.1 - Update a boid's velocity using the uniform grid to reduce
  // the number of boids that need to be checked.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (index < N) {
		// Identify the grid cell that this particle is in.
		glm::vec3 myPos = pos[index];
		glm::vec3 myGridPos = myPos - gridMin;
		myGridPos *= inverseCellWidth;     // Just like in kernComputeIndices()
		myGridPos = glm::floor(myGridPos); // Just like in kernComputeIndices()

		// Identify which cells contain neighbors
		// Want to find each grid where one of our rules can apply. Therefore, we need to know the distance of our rules.
		float neighbor_distance = glm::max(glm::max(rule1Distance, rule2Distance), rule3Distance);
		// Create a vector with this value.
		glm::vec3 neighbor_distance_vec(neighbor_distance);
		// Remember, everything is a cube! If we check the corners of the cube, we will know
		// what the rest of the cube looks like. ie, if min is {0, 0, 0} and max is {1, 1, 1},
		// we know where each of the other corners lie. (Also true for rectangluar cubes)
		glm::vec3 min_neighbor_cell = glm::floor((myPos - gridMin - neighbor_distance) * inverseCellWidth);
		glm::vec3 max_neighbor_cell = glm::floor((myPos - gridMin + neighbor_distance) * inverseCellWidth);
		// Now, if myPos is situated on any of the axis of the cell, then min and/or max may not change.
		// This is clear from the case where myPos is in the middle of the cell. In that case, if the cellWidth
		// is equal to the neighbor_disance, then the cube will consist of only the cell.
		// So we don't need to check the cell interior edge conditions, those are covered already.

		// Issue: What about the edge of the grid? We need to wrap around.
		// We can handle this during the search. If any cell value exceeds the resolution of the grid,
		// then we loop around.
		glm::vec3 min_cell_search = glm::clamp(min_neighbor_cell, glm::vec3(0), glm::vec3(gridResolution));
		glm::vec3 max_cell_search = glm::clamp(max_neighbor_cell, glm::vec3(0), glm::vec3(gridResolution));

		// After all that work, we now start applying rules! Instead of searching over N boids, we will search over the boids
		// in each cell between min_cell_search and max_cell_search only.
		// I can already see how making the boids cohenernt in memory helps simplify this, but that's for later.
		glm::vec3 velocity_change(0.0f);
		glm::vec3 selfPos = pos[index];
		glm::vec3 selfVelocity = vel1[index];
		glm::vec3 alignment_perceived_velocity(0.0f);
		glm::vec3 cohesion_perceived_center(0.0f);
		glm::vec3 seperation(0.0f);
		int alignment_neighbors = 0.0;
		int cohesion_neighbors = 0.0;
		int seperation_neighbors = 0.0;

		for (float z = min_cell_search.z; z <= max_cell_search.z; z++) {
			for (float y = min_cell_search.y; y <= max_cell_search.y; y++) {
				for (float x = min_cell_search.x; x <= max_cell_search.x; x++) {
					// Yikes! Triple for loop??? Not that bad, min and max differ by at most one.
					// Calculate grid index of the grid under inspection
					int gridIdx = gridIndex3Dto1D(x, y, z, gridResolution);

					// Get the start and end indices for the boids
					int b_start = gridCellStartIndices[gridIdx];
					int b_end   = gridCellEndIndices[gridIdx];

					// Check if any values are -1, meaning an empty cell.
					// Also check if less than N, don't want to cause a segfault.
					if (b_start < 0 || b_start > N || b_end < 0 || b_end > N) {
						continue;
					}

					// We now have the boids we need. Run each rule over the range of boids.
					// WOW: Unrolling these calls gave a 2.27X performance boost!!!! Cache benefits?
					//velocity_change += boidAlignmentRuleGrid(N, index, particleArrayIndices, b_start, b_end, pos, vel1);
					//velocity_change += boidCohesionRuleGrid(N, index, particleArrayIndices, b_start, b_end, pos, vel1);
					//velocity_change += boidSeperationRuleGrid(N, index, particleArrayIndices, b_start, b_end, pos, vel1);
					
					for (int i = b_start; i <= b_end; i++) {
						int boid_idx = particleArrayIndices[i];
						if (index == boid_idx) { // Dip out early.
							continue;
						}

						// Get relevant data
						glm::vec3 boid_pos = pos[boid_idx];
						float distance = glm::distance(selfPos, boid_pos);

						// Cohesion
						if (distance < rule1Distance) {
							cohesion_perceived_center += boid_pos;
							cohesion_neighbors++;
						}
						// Seperation
						if (distance < rule2Distance) {
							seperation -= boid_pos - selfPos;
							seperation_neighbors++;
						}
						// Alignment
						if (distance < rule3Distance) {
							alignment_perceived_velocity += vel1[boid_idx];
							alignment_neighbors++;
						}
					}
				}
			}
		}

		// Finalize Cohesion values
		if (cohesion_neighbors) {
			cohesion_perceived_center /= cohesion_neighbors;
			velocity_change += (cohesion_perceived_center - selfPos) * rule1Scale;
		}
		// Finalize Seperation Values
		if (seperation_neighbors) {
			velocity_change += seperation * rule2Scale;
		}
		// Finalize Alignment Values
		if (alignment_neighbors) {
			alignment_perceived_velocity /= alignment_neighbors;
			velocity_change += alignment_perceived_velocity * rule3Scale;
		}

		// Calculated total velocity change! Now apply, clamp, and store.
		glm::vec3 newV = vel1[index] + velocity_change;
		if (glm::length(newV) > maxSpeed) {
			newV = glm::normalize(newV) * maxSpeed;
		}
		vel2[index] = newV;
	}
}

__global__ void kernUpdateVelNeighborSearchCoherent(
  int N, int gridResolution, glm::vec3 gridMin,
  float inverseCellWidth, float cellWidth,
  int *gridCellStartIndices, int *gridCellEndIndices,
  glm::vec3 *pos, glm::vec3 *vel1, glm::vec3 *vel2) {
  // TODO-2.3 - This should be very similar to kernUpdateVelNeighborSearchScattered,
  // except with one less level of indirection.
  // This should expect gridCellStartIndices and gridCellEndIndices to refer
  // directly to pos and vel1.
  // - Identify the grid cell that this particle is in
  // - Identify which cells may contain neighbors. This isn't always 8.
  // - For each cell, read the start/end indices in the boid pointer array.
  //   DIFFERENCE: For best results, consider what order the cells should be
  //   checked in to maximize the memory benefits of reordering the boids data.
  // - Access each boid in the cell and compute velocity change from
  //   the boids rules, if this boid is within the neighborhood distance.
  // - Clamp the speed change before putting the new speed in vel2
}

/**
* Step the entire N-body simulation by `dt` seconds.
*/
void Boids::stepSimulationNaive(float dt) {
	// TODO-1.2 - use the kernels you wrote to step the simulation forward in time.
	// TODO-1.2 ping-pong the velocity buffers

	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);
	kernUpdateVelocityBruteForce<<<fullBlocksPerGrid, blockSize>>>(numObjects, dev_pos, dev_vel1, dev_vel2);
	kernUpdatePos<<<fullBlocksPerGrid, blockSize>>>(numObjects, dt, dev_pos, dev_vel2); // Use new velocity!

	// Swap buffers, ping pong!
	std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationScatteredGrid(float dt) {
  // TODO-2.1
  // Uniform Grid Neighbor search using Thrust sort.
  // In Parallel:
  // - label each particle with its array index as well as its grid index.
  //   Use 2x width grids.
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed

	dim3 fullBlocksPerGrid((numObjects + blockSize - 1) / blockSize);

	// Run the index labeling kernel
	kernComputeIndices<<<fullBlocksPerGrid, blockSize >>>(
		numObjects,
		gridSideCount,
		gridMinimum,
		gridInverseCellWidth,
		dev_pos,
		dev_particleArrayIndices,
		dev_particleGridIndices
	);
	checkCUDAErrorWithLine("kernComputeIndices failed!");

	// Use Thrust API to sort the indicies...
	thrust::sort_by_key(
		dev_thrust_particleGridIndices,
		dev_thrust_particleGridIndices + numObjects,
		dev_thrust_particleArrayIndices
	);

	// Locate start and stop indicies
	kernResetIntBuffer<<<fullBlocksPerGrid, blockSize >>>(gridCellCount, dev_gridCellStartIndices, -1);
	checkCUDAErrorWithLine("kernResetIntBuffer1 failed!");
	kernResetIntBuffer<<<fullBlocksPerGrid, blockSize >>>(gridCellCount, dev_gridCellEndIndices,   -1);
	checkCUDAErrorWithLine("kernResetIntBuffer2 failed!");
	kernIdentifyCellStartEnd<<<fullBlocksPerGrid, blockSize >>> (
		numObjects,
		dev_particleGridIndices,
		dev_gridCellStartIndices,
		dev_gridCellEndIndices
	);
	checkCUDAErrorWithLine("kernIdentifyCellStartEnd failed!");

	// The fun part! Calculate velocity changes
	kernUpdateVelNeighborSearchScattered<<<fullBlocksPerGrid, blockSize >>>(
		numObjects,
		gridSideCount,
		gridMinimum,
		gridInverseCellWidth,
		gridCellWidth,
		dev_gridCellStartIndices,
		dev_gridCellEndIndices,
		dev_particleArrayIndices,
		dev_pos,
		dev_vel1,
		dev_vel2
	);
	checkCUDAErrorWithLine("kernUpdateVelNeighborSearchScattered failed!");

	// Update positions
	kernUpdatePos<<<fullBlocksPerGrid, blockSize >>>(
		numObjects,
		dt,
		dev_pos,
		dev_vel1
	);
	checkCUDAErrorWithLine("kernUpdatePos failed!");

	// Swap buffers and you're done!
	std::swap(dev_vel1, dev_vel2);
}

void Boids::stepSimulationCoherentGrid(float dt) {
  // TODO-2.3 - start by copying Boids::stepSimulationNaiveGrid
  // Uniform Grid Neighbor search using Thrust sort on cell-coherent data.
  // In Parallel:
  // - Label each particle with its array index as well as its grid index.
  //   Use 2x width grids
  // - Unstable key sort using Thrust. A stable sort isn't necessary, but you
  //   are welcome to do a performance comparison.
  // - Naively unroll the loop for finding the start and end indices of each
  //   cell's data pointers in the array of boid indices
  // - BIG DIFFERENCE: use the rearranged array index buffer to reshuffle all
  //   the particle data in the simulation array.
  //   CONSIDER WHAT ADDITIONAL BUFFERS YOU NEED
  // - Perform velocity updates using neighbor search
  // - Update positions
  // - Ping-pong buffers as needed. THIS MAY BE DIFFERENT FROM BEFORE.
}

void Boids::endSimulation() {
  cudaFree(dev_vel1);
  cudaFree(dev_vel2);
  cudaFree(dev_pos);

  // TODO-2.1 TODO-2.3 - Free any additional buffers here.
  cudaFree(dev_particleArrayIndices);
  cudaFree(dev_particleGridIndices);
  cudaFree(dev_gridCellStartIndices);
  cudaFree(dev_gridCellEndIndices);
}

void Boids::unitTest() {
  // LOOK-1.2 Feel free to write additional tests here.

  // test unstable sort
  int *dev_intKeys;
  int *dev_intValues;
  int N = 10;

  std::unique_ptr<int[]>intKeys{ new int[N] };
  std::unique_ptr<int[]>intValues{ new int[N] };

  intKeys[0] = 0; intValues[0] = 0;
  intKeys[1] = 1; intValues[1] = 1;
  intKeys[2] = 0; intValues[2] = 2;
  intKeys[3] = 3; intValues[3] = 3;
  intKeys[4] = 0; intValues[4] = 4;
  intKeys[5] = 2; intValues[5] = 5;
  intKeys[6] = 2; intValues[6] = 6;
  intKeys[7] = 0; intValues[7] = 7;
  intKeys[8] = 5; intValues[8] = 8;
  intKeys[9] = 6; intValues[9] = 9;

  cudaMalloc((void**)&dev_intKeys, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intKeys failed!");

  cudaMalloc((void**)&dev_intValues, N * sizeof(int));
  checkCUDAErrorWithLine("cudaMalloc dev_intValues failed!");

  dim3 fullBlocksPerGrid((N + blockSize - 1) / blockSize);

  std::cout << "before unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // How to copy data to the GPU
  cudaMemcpy(dev_intKeys, intKeys.get(), sizeof(int) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(dev_intValues, intValues.get(), sizeof(int) * N, cudaMemcpyHostToDevice);

  // Wrap device vectors in thrust iterators for use with thrust.
  thrust::device_ptr<int> dev_thrust_keys(dev_intKeys);
  thrust::device_ptr<int> dev_thrust_values(dev_intValues);
  // LOOK-2.1 Example for using thrust::sort_by_key
  thrust::sort_by_key(dev_thrust_keys, dev_thrust_keys + N, dev_thrust_values);

  // How to copy data back to the CPU side from the GPU
  cudaMemcpy(intKeys.get(), dev_intKeys, sizeof(int) * N, cudaMemcpyDeviceToHost);
  cudaMemcpy(intValues.get(), dev_intValues, sizeof(int) * N, cudaMemcpyDeviceToHost);
  checkCUDAErrorWithLine("memcpy back failed!");

  std::cout << "after unstable sort: " << std::endl;
  for (int i = 0; i < N; i++) {
    std::cout << "  key: " << intKeys[i];
    std::cout << " value: " << intValues[i] << std::endl;
  }

  // cleanup
  cudaFree(dev_intKeys);
  cudaFree(dev_intValues);
  checkCUDAErrorWithLine("cudaFree failed!");
  return;
}
