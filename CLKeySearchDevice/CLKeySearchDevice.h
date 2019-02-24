#ifndef _CL_KEYSEARCH_DEVICE_H
#define _CL_KEYSEARCH_DEVICE_H

#include "KeySearchDevice.h"
#include "clContext.h"

typedef struct CLTargetList_
{
    cl_ulong mask = 0;
    cl_ulong size = 0;
    cl_mem ptr = 0;
}CLTargetList;

class CLKeySearchDevice : public KeySearchDevice {

private:
    cl::CLContext *_clContext = NULL;
    cl::CLProgram *_clProgram = NULL;
    cl::CLKernel *_initKeysKernel = NULL;
    cl::CLKernel *_stepKernel = NULL;
    cl::CLKernel *_stepKernelWithDouble = NULL;

    uint64_t _globalMemSize = 0;
    uint64_t _pointsMemSize = 0;
    uint64_t _targetMemSize = 0;

    CLTargetList _deviceTargetList;

    secp256k1::uint256 _start;
    secp256k1::uint256 _end;
    
    std::vector<hash160> _targetList;

    std::vector<KeySearchResult> _results;

    int _blocks;

    int _threads;

    int _pointsPerThread;

    cl_device_id _device;

    int _compression = PointCompressionType::COMPRESSED;
    
    bool _randomMode = false;

    uint64_t _iterations = 0;

    secp256k1::uint256 _stride = 1;

    std::string _deviceName;

    // Device memory pointers
    cl_mem _chain = NULL;

    cl_mem _x = NULL;

    cl_mem _y = NULL;

    cl_mem _xInc = NULL;

    cl_mem _yInc = NULL;

    cl_mem _privateKeys = NULL;

    cl_mem _xTable = NULL;
    
    cl_mem _yTable = NULL;

    cl_mem _deviceResults = NULL;

    cl_mem _deviceResultsCount = NULL;

    cl_mem _targets = NULL;

    void generateStartingPoints();

    void setIncrementor(secp256k1::ecpoint &p);

    void splatBigInt(secp256k1::uint256 &k, unsigned int *ptr);

    void allocateBuffers();

    void initializeBasePoints();

    int getIndex(int block, int thread, int idx);

    void splatBigInt(unsigned int *dest, int block, int thread, int idx, const secp256k1::uint256 &i);
    secp256k1::uint256 readBigInt(unsigned int *src, int block, int thread, int idx);

    void selfTest();

    bool _useBloomFilter = false;

    void setTargetsInternal();
    void setTargetsList();
    void setBloomFilter();

    void getResultsInternal();

    bool isTargetInList(const unsigned int hash[5]);

    void removeTargetFromList(const unsigned int hash[5]);

    uint32_t getPrivateKeyOffset(int thread, int block, int idx);

    void initializeBloomFilter(const std::vector<struct hash160> &targets, uint64_t mask);

    uint64_t getOptimalBloomFilterMask(double p, size_t n);

    std::vector<secp256k1::uint256> exponents;

public:

    CLKeySearchDevice(uint64_t device, int threads, int pointsPerThread, int blocks = 0);
    ~CLKeySearchDevice();


    // Initialize the device
    virtual void init(const secp256k1::uint256 &start, const secp256k1::uint256 &end, int compression, const secp256k1::uint256 &stride, bool randomMode);

    // Perform one iteration
    virtual void doStep();

    // Tell the device which addresses to search for
    virtual void setTargets(const std::set<KeySearchTarget> &targets);

    // Get the private keys that have been found so far
    virtual size_t getResults(std::vector<KeySearchResult> &results);

    // The number of keys searched at each step
    virtual uint64_t keysPerStep();

    // The name of the device
    virtual std::string getDeviceName();

    // Memory information for this device
    virtual void getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem);

    virtual secp256k1::uint256 getNextKey();
};

#endif

