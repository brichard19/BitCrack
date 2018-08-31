#include "cudabridge.h"


__global__ void keyFinderKernel(int points, int compression);
__global__ void keyFinderKernelWithDouble(int points, int compression);

void callKeyFinderKernel(KernelParams &params, bool useDouble, int compression)
{
	if(useDouble) {
		keyFinderKernelWithDouble <<<params.blocks, params.threads >> >(params.points, compression);
	} else {
		keyFinderKernel <<<params.blocks, params.threads >> > (params.points, compression);
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