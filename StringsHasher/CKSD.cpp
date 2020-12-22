#include "CKSD.h"
#include "Logger.h"
#include "util.h"
#include "cudabridge.h"
#include "AddressUtil.h"

void CKSD::cudaCall(cudaError_t err)
{
    if(err) {
        std::string errStr = cudaGetErrorString(err);

        throw KeySearchException(errStr);
    }
}

CKSD::CKSD(int device, int threads, int pointsPerThread, int blocks)
{
	
    _device = device;

    _pointsPerThread = pointsPerThread;
}

void CKSD::init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride)
{
    if(start.cmp(secp256k1::N) >= 0) {
        throw KeySearchException("Starting key is out of range");
    }

    _compression = compression;

    cudaCall(cudaSetDevice(_device));

    // Block on kernel calls
    cudaCall(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));

    // Use a larger portion of shared memory for L1 cache
    cudaCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));


}
