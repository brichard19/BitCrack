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

class CudaKeySearchDevice : public KeySearchDevice {

private:

    int _device;

    int _blocks;

    int _threads;

    int _pointsPerThread;

    int _compression;

    std::vector<KeySearchResult> _results;

    std::string _deviceName;

    secp256k1::uint256 _startExponent;

    uint64_t _iterations;

    void cudaCall(cudaError_t err);

    void generateStartingPoints();

    CudaDeviceKeys _deviceKeys;

    CudaAtomicList _resultList;

    CudaHashLookup _targetLookup;

    void getResultsInternal();

    std::vector<hash160> _targets;

    bool isTargetInList(const unsigned int hash[5]);
    
    void removeTargetFromList(const unsigned int hash[5]);

    uint32_t getPrivateKeyOffset(int thread, int block, int point);

    uint64_t _stride;

    bool verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5], bool compressed);

public:

    CudaKeySearchDevice(int device, int threads, int pointsPerThread, int blocks = 0);

    virtual void init(const secp256k1::uint256 &start, int compression, uint64_t stride);

    virtual void doStep();

    virtual void setTargets(const std::set<KeySearchTarget> &targets);

    virtual size_t getResults(std::vector<KeySearchResult> &results);

    virtual uint64_t keysPerStep();

    virtual std::string getDeviceName();

    virtual void getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem);

    virtual secp256k1::uint256 getNextKey();
};

#endif