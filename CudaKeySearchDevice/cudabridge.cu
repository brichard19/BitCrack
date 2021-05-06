#include "cudabridge.h"


__global__ void keyCheckKernel(unsigned int *xPtr, unsigned int *yPtr, int blocks, int compression);
__global__ void keyFinderKernel(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chainPtr, int points);
__global__ void keyFinderKernelWithDouble(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chainPtr, int points);

void callKeyFinderKernel(int blocks, int threads, int points, unsigned int *xPtr, unsigned int *yPtr, unsigned int *chainPtr, bool useDouble, int compression)
{
    int blocksMultPoints = blocks * points;

    keyCheckKernel <<<blocksMultPoints, threads>>> (xPtr, yPtr, blocks, compression);
    checkKernelLaunch();

    if(useDouble) {
        keyFinderKernelWithDouble <<<blocks, threads >>> (xPtr, yPtr, chainPtr, points);
    } else {
        keyFinderKernel <<<blocks, threads>>> (xPtr, yPtr, chainPtr, points);
    }
    waitForKernel();
}

void checkKernelLaunch()
{
    // Check for kernel launch error
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        throw cuda::CudaException(err);
    }
}

void waitForKernel()
{
    // Check for kernel launch error
    cudaError_t err = cudaGetLastError();

    if(err != cudaSuccess) {
        throw cuda::CudaException(err);
    }
 
    // Wait for kernel to complete
    err = cudaDeviceSynchronize();
	fflush(stdout);
	if(err != cudaSuccess) {
		throw cuda::CudaException(err);
	}
}
