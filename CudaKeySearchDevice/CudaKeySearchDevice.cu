#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "KeySearchTypes.h"
#include "CudaKeySearchDevice.h"

#include "ptx.cuh"
#include "secp256k1.cuh"

#include "secp256k1.h"

#include "CudaHashLookup.cuh"
#include "CudaAtomicList.cuh"
#include "CudaDeviceKeys.cuh"

__constant__ unsigned int _INC_X[8];
__constant__ unsigned int _INC_Y[8];


/**
 *Sets the EC point which all points will be incremented by
 */
cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y)
{
    unsigned int xWords[8];
    unsigned int yWords[8];

    x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
    y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

    cudaError_t err = cudaMemcpyToSymbol(_INC_X, xWords, sizeof(unsigned int) * 8);
    if(err) {
        return err;
    }

    return cudaMemcpyToSymbol(_INC_Y, yWords, sizeof(unsigned int) * 8);
}

__device__ void doIteration(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chain, int pointsPerThread)
{
    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];
        readInt(xPtr, i, x);

        beginBatchAdd(_INC_X, x, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAdd(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}

__device__ void doIterationWithDouble(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chain, int pointsPerThread)
{
    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];
        readInt(xPtr, i, x);

        beginBatchAddWithDouble(_INC_X, _INC_Y, xPtr, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAddWithDouble(_INC_X, _INC_Y, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}

/**
* Performs a single iteration
*/
__global__ void keyFinderKernel(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chainPtr, int points)
{
    doIteration(xPtr, yPtr, chainPtr, points);
}

__global__ void keyFinderKernelWithDouble(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chainPtr, int points)
{
    doIterationWithDouble(xPtr, yPtr, chainPtr, points);
}
