#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "SkeinTest.h"

#define TILE_SIZE 256

// TODO signature
// ...UBI kernel? Tree level kernel?
__global__ void SkeinTreeKernel( ) {

	int tx = threadIdx.x; // Thread index
	int bx = blockIdx.x;  // Block index
	int index = bx * TILE_SIZE + tx; // Global thread index

	// TODO I have no idea what I'm doing

}

// TODO signature?
int SkeinTreeHash_GPU( uint_t blkSize, uint_t hashBits, const u08b_t *msg, size_t msgBytes,
	uint_t leaf, uint_t node, uint_t maxLevel, u08b_t *hashRes ) {

	// TODO seriously I have no idea what I'm doing

	// Make sure everything worked okay; if not, indicate that error occurred
	cudaError_t error = cudaGetLastError();
	if(error) {
		printf("ERROR: %s\n", cudaGetErrorString(error));
		return 1;
	}

	return 0;

}