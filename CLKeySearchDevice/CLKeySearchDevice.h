#ifndef CL_KEYSEARCH_DEVICE_H
#define CL_KEYSEARCH_DEVICE_H

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
    uint64_t _bufferMemSize = 0;
    uint64_t _targetMemSize = 0;

    CLTargetList _deviceTargetList;

    secp256k1::uint256 _start;
    
    std::vector<hash160> _targetList;

    std::vector<KeySearchResult> _results;

    int _blocks;

    int _threads;

    int _points;

    cl_device_id _device;

    int _compression = PointCompressionType::COMPRESSED;

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

    void splatBigInt(unsigned int *dest, int idx, secp256k1::uint256 &k);
    secp256k1::uint256 readBigInt(unsigned int *src, int idx);

    bool _useBloomFilter = false;

    void setTargetsInternal();
    void setTargetsList();
    void setBloomFilter();

    void getResultsInternal();

    bool isTargetInList(const unsigned int hash[5]);

    void removeTargetFromList(const unsigned int hash[5]);

    void initializeBloomFilter(const std::vector<struct hash160> &targets, uint64_t mask);

    uint64_t getOptimalBloomFilterMask(double p, size_t n);

public:

    CLKeySearchDevice(uint64_t device, int threads, int pointsPerThread, int blocks = 0, int compressionMode = PointCompressionType::COMPRESSED);
    ~CLKeySearchDevice();


    // Initialize the device
    virtual void init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride);

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
