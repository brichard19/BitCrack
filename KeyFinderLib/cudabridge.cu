#include "cudabridge.h"


__global__ void keyFinderKernel(int points, unsigned int *x, unsigned int *y, unsigned int *chain, unsigned int *numResults, void *results, int compression);
__global__ void keyFinderKernelWithDouble(int points, unsigned int *x, unsigned int *y, unsigned int *chain, unsigned int *numResults, void *results, int compression);

void callKeyFinderKernel(KernelParams &params, bool useDouble, int compression)
{
	if(useDouble) {
		keyFinderKernelWithDouble <<<params.blocks, params.threads >> >(params.points, params.x, params.y, params.chain, params.numResults, params.results, compression);
	} else {
		keyFinderKernel <<<params.blocks, params.threads >> > (params.points, params.x, params.y, params.chain, params.numResults, params.results, compression);
	}
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