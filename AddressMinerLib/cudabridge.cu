#include "cudabridge.h"


__global__ void addressMinerKernel(int points, unsigned int flags, unsigned int *x, unsigned int *y, unsigned int *chain, unsigned int *numResults, void *results);


void callAddressMinerKernel(KernelParams &params)
{
	addressMinerKernel<<<params.blocks, params.threads>>>(params.points, params.flags, params.x, params.y, params.chain, params.numResults, params.results);
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