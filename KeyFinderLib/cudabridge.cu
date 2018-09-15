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
    // Check for kernel launch error
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        printf("Run time error\n");
        throw cuda::CudaException(err);
    }
 
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
	fflush(stdout);
	if(err != cudaSuccess) {
        printf("Kernel error\n");
		throw cuda::CudaException(err);
	}
}