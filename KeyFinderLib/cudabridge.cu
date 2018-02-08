#include "cudabridge.h"


__global__ void keyFinderKernel(int points, unsigned int flags, unsigned int *x, unsigned int *y, unsigned int *chain, unsigned int *numResults, void *results, bool useDouble);


void callKeyFinderKernel(KernelParams &params, bool useDouble)
{
	keyFinderKernel<<<params.blocks, params.threads>>>(params.points, params.flags, params.x, params.y, params.chain, params.numResults, params.results, useDouble);
	waitForKernel();
}


void waitForKernel()
{
	// Wait for kernel to complete
	cudaError_t err = cudaDeviceSynchronize();
	fflush(stdout);
	if(err != cudaSuccess) {
		throw CudaException(err);
	}
}