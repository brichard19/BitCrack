#ifndef _CUDA_KEY_SEARCH_DEVICE
#define _CUDA_KEY_SEARCH_DEVICE

#include "KeySearchDevice.h"
#include <vector>
#include <cuda_runtime.h>
#include "secp256k1.h"
#include "CudaDeviceKeys.h"
#include "CudaHashLookup.h"
#include "CudaAtomicList.h"
#include "cudaUtil.h"

// Structures that exist on both host and device side
struct CudaDeviceResult {
    int thread;
    int block;
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
};

class CKSD : public KeySearchDevice {

private:

    int _device;

    int _blocks;

    int _threads;

    int _pointsPerThread;

    int _compression;

    std::string _deviceName;

    void cudaCall(cudaError_t err);


public:

	CKSD(int device, int threads, int pointsPerThread, int blocks = 0);

    virtual void init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride);

};

#endif