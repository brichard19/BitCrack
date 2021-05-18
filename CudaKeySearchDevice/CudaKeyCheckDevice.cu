#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "KeySearchTypes.h"
#include "CudaKeySearchDevice.h"
#include "ptx.cuh"
#include "secp256k1.cuh"

#include "sha256.cuh"
#include "ripemd160.cuh"

#include "secp256k1.h"

#include "CudaHashLookup.cuh"
#include "CudaAtomicList.cuh"
#include "CudaDeviceKeys.cuh"


__device__ void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    const unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for(int i = 0; i < 5; i++) {
        hOut[i] = endian(hIn[i] + iv[(i + 1) % 5]);
    }
}

__device__ void hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKey(x, y, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

__device__ void hashPublicKeyCompressed(const unsigned int *x, unsigned int yParity, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x, yParity, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

__device__ void setResultFound(const int numBlocks, bool compressed, unsigned int x[8], unsigned int y[8], unsigned int digest[5])
{
    CudaDeviceResult r;

    unsigned int virtualPointsPerThread = blockIdx.x / numBlocks; // ex: (uint) 8 / 6 = 1
    unsigned int virtualBlock = blockIdx.x % numBlocks;

    r.block = virtualBlock;
    r.thread = threadIdx.x;
    r.idx = virtualPointsPerThread;
    r.compressed = compressed;

    for(int i = 0; i < 8; i++) {
        r.x[i] = x[i];
        r.y[i] = y[i];
    }

    doRMD160FinalRound(digest, r.digest);

    atomicListAdd(&r, sizeof(r));
}

/**
 * Reads an 2-vector4 big integer from device memory with virtualized loop
 */
__device__ static void readIntVirtualized(unsigned int *ara, const int numBlocks, unsigned int x[8])
{
    uint4 *araTmp = reinterpret_cast<uint4 *>(ara);

    unsigned int virtualPointsPerThread = blockIdx.x / numBlocks; // ex: (uint) 8 / 6 = 1
    unsigned int virtualBlock = blockIdx.x % numBlocks; // ex: 8 % 6 = 2
    unsigned int totalThreads = numBlocks * blockDim.x; // 6 * 128 = 768
    unsigned int base = virtualPointsPerThread * totalThreads * 2; // 1 * 768
    unsigned int threadId = blockDim.x * virtualBlock + threadIdx.x; // 128 * 2 * ... 
    unsigned int index = base + threadId;

    uint4 xTmp = araTmp[index];
    x[0] = xTmp.x;
    x[1] = xTmp.y;
    x[2] = xTmp.z;
    x[3] = xTmp.w;

    index += totalThreads;

    xTmp = araTmp[index];
    x[4] = xTmp.x;
    x[5] = xTmp.y;
    x[6] = xTmp.z;
    x[7] = xTmp.w;
}

__device__ static unsigned int readIntLSWVirtualized(unsigned int *ara, const int numBlocks)
{
    uint4 *araTmp = reinterpret_cast<uint4 *>(ara);

    unsigned int virtualPointsPerThread = blockIdx.x / numBlocks; // ex: (uint) 8 / 6 = 1
    unsigned int virtualBlock = blockIdx.x % numBlocks; // ex: 8 % 6 = 2
    unsigned int totalThreads = numBlocks * blockDim.x; // 6 * 128 = 768
    unsigned int base = virtualPointsPerThread * totalThreads * 2; // 1 * 768
    unsigned int threadId = blockDim.x * virtualBlock + threadIdx.x; // 128 * 2 * ... 
    unsigned int index = base + threadId;

    index += totalThreads;

    uint4 xTmp = araTmp[index];

    return xTmp.w;
}

__device__ void hashAndCheck(unsigned int *xPtr, unsigned int *yPtr, const int numBlocks, const int compression)
{
    unsigned int x[8];
    unsigned int digest[5];

    readIntVirtualized(xPtr, numBlocks, x);

    if(compression == PointCompressionType::UNCOMPRESSED || compression == PointCompressionType::BOTH) {
        unsigned int y[8];
        readIntVirtualized(yPtr, numBlocks, y);

        hashPublicKey(x, y, digest);

        if(checkHash(digest)) {
            setResultFound(numBlocks, false, x, y, digest);
        }
    }

    if(compression == PointCompressionType::COMPRESSED || compression == PointCompressionType::BOTH) {
        hashPublicKeyCompressed(x, readIntLSWVirtualized(yPtr, numBlocks), digest);

        if(checkHash(digest)) {
            unsigned int y[8];
            readIntVirtualized(yPtr, numBlocks, y);
            setResultFound(numBlocks, true, x, y, digest);
        }
    }
}

/**
* Performs a single iteration
*/
__global__ void keyCheckKernel(unsigned int *xPtr, unsigned int *yPtr, const int blocks, const int compression)
{
    hashAndCheck(xPtr, yPtr, blocks, compression);
}

