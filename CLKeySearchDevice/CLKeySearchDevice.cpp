#include <cmath>
#include "Logger.h"
#include "util.h"
#include "CLKeySearchDevice.h"

// Defined in bitcrack_cl.cpp which gets build in the pre-build event
extern char _bitcrack_cl[];

typedef struct {
    int thread;
    int block;
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
}CLDeviceResult;


static void undoRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for(int i = 0; i < 5; i++) {
        hOut[i] = util::endian(hIn[i]) - iv[(i + 1) % 5];
    }
}

CLKeySearchDevice::CLKeySearchDevice(uint64_t device, int threads, int pointsPerThread, int blocks)
{
    _threads = threads;
    _blocks = blocks;
    _pointsPerThread = pointsPerThread;
    _device = (cl_device_id)device;


    if(threads <= 0 || threads % 32 != 0) {
        throw KeySearchException("The number of threads must be a multiple of 32");
    }

    if(pointsPerThread <= 0) {
        throw KeySearchException("At least 1 point per thread required");
    }

    try {
        // Create the context
        _clContext = new cl::CLContext(_device);
        Logger::log(LogLevel::Info, "Compiling OpenCL kernels...");
        //_clProgram = new cl::CLProgram(*_clContext, util::getExeDirectory() + "KeySearch.cl");
        _clProgram = new cl::CLProgram(*_clContext, _bitcrack_cl);

        // Load the kernels
        _initKeysKernel = new cl::CLKernel(*_clProgram, "multiplyStepKernel");
        _initKeysKernel->getWorkGroupSize();

        _stepKernel = new cl::CLKernel(*_clProgram, "keyFinderKernel");
        _stepKernelWithDouble = new cl::CLKernel(*_clProgram, "keyFinderKernelWithDouble");

        _globalMemSize = _clContext->getGlobalMemorySize();

        _deviceName = _clContext->getDeviceName();
    } catch(cl::CLException ex) {
        throw KeySearchException(ex.msg);
    }

    _iterations = 0;
}

CLKeySearchDevice::~CLKeySearchDevice()
{
    _clContext->free(_x);
    _clContext->free(_y);
    _clContext->free(_xTable);
    _clContext->free(_yTable);
    _clContext->free(_xInc);
    _clContext->free(_yInc);
    _clContext->free(_deviceResults);
    _clContext->free(_deviceResultsCount);

    delete _stepKernel;
    delete _stepKernelWithDouble;
    delete _initKeysKernel;
    delete _clContext;
}

uint64_t CLKeySearchDevice::getOptimalBloomFilterMask(double p, size_t n)
{
    double m = 3.6 * ceil((n * std::log(p)) / std::log(1 / std::pow(2, std::log(2))));

    unsigned int bits = std::ceil(std::log(m) / std::log(2));

    return ((uint64_t)1 << bits) - 1;
}

void CLKeySearchDevice::initializeBloomFilter(const std::vector<struct hash160> &targets, uint64_t mask)
{
    size_t sizeInWords = (mask + 1) / 32;

    uint32_t *buf = new uint32_t[sizeInWords];

    for(size_t i = 0; i < sizeInWords; i++) {
        buf[i] = 0;
    }

    for(unsigned int k = 0; k < targets.size(); k++) {

        unsigned int hash[5];
        unsigned int h5 = 0;

        uint64_t idx[5];

        undoRMD160FinalRound(targets[k].h, hash);

        for(int i = 0; i < 5; i++) {
            h5 += hash[i];
        }

        idx[0] = ((hash[0] << 6) | (h5 & 0x3f)) & mask;
        idx[1] = ((hash[1] << 6) | ((h5 >> 6) & 0x3f)) & mask;
        idx[2] = ((hash[2] << 6) | ((h5 >> 12) & 0x3f)) & mask;
        idx[3] = ((hash[3] << 6) | ((h5 >> 18) & 0x3f)) & mask;
        idx[4] = ((hash[4] << 6) | ((h5 >> 24) & 0x3f)) & mask;

        for(int i = 0; i < 5; i++) {
            uint64_t j = idx[i];
            buf[j / 32] |= 1 << (j % 32);
        }
    }


    _targetMemSize = sizeInWords * sizeof(uint32_t);

    _deviceTargetList.mask = mask;
    _deviceTargetList.ptr = _clContext->malloc(sizeInWords * sizeof(uint32_t));
    _deviceTargetList.size = targets.size();
    _clContext->copyHostToDevice(buf, _deviceTargetList.ptr, sizeInWords * sizeof(uint32_t));

    delete[] buf;
}

void CLKeySearchDevice::allocateBuffers()
{
    size_t numKeys = _threads * _blocks * _pointsPerThread;
    size_t size = numKeys * 8 * sizeof(unsigned int);

    // X values
    _x = _clContext->malloc(size);
    _clContext->memset(_x, -1, size);

    // Y values
    _y = _clContext->malloc(size);
    _clContext->memset(_y, -1, size);

    // Multiplicaiton chain for batch inverse
    _chain = _clContext->malloc(size);

    // Private keys for initialization
    _privateKeys = _clContext->malloc(size, CL_MEM_READ_ONLY);

    // Lookup table for initialization
    _xTable = _clContext->malloc(256 * 8 * sizeof(unsigned int), CL_MEM_READ_ONLY);
    _yTable = _clContext->malloc(256 * 8 * sizeof(unsigned int), CL_MEM_READ_ONLY);

    // Value to increment by
    _xInc = _clContext->malloc(8 * sizeof(unsigned int), CL_MEM_READ_ONLY);
    _yInc = _clContext->malloc(8 * sizeof(unsigned int), CL_MEM_READ_ONLY);

    // Buffer for storing results
    _deviceResults = _clContext->malloc(128 * sizeof(CLDeviceResult));
    _deviceResultsCount = _clContext->malloc(sizeof(unsigned int));
}

void CLKeySearchDevice::setIncrementor(secp256k1::ecpoint &p)
{
    unsigned int buf[8];

    p.x.exportWords(buf, 8, secp256k1::uint256::BigEndian);
    _clContext->copyHostToDevice(buf, _xInc, 8 * sizeof(unsigned int));

    p.y.exportWords(buf, 8, secp256k1::uint256::BigEndian);
    _clContext->copyHostToDevice(buf, _yInc, 8 * sizeof(unsigned int));
}

void CLKeySearchDevice::init(const secp256k1::uint256 &start, int compression)
{
    if(start.cmp(secp256k1::N) >= 0) {
        throw KeySearchException("Starting key is out of range");
    }

    _start = start;

    _compression = compression;

    allocateBuffers();

    generateStartingPoints();

    // Set the incrementor
    secp256k1::ecpoint g = secp256k1::G();
    secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256(_threads * _blocks * _pointsPerThread), g);

    setIncrementor(p);
}

void CLKeySearchDevice::doStep()
{
    try {
        uint64_t numKeys = _blocks * _threads * _pointsPerThread;

        if(_iterations < 2 && _start.cmp(numKeys) <= 0) {

            _stepKernelWithDouble->call(
                _blocks,
                _threads,
                _pointsPerThread,
                _compression,
                _chain,
                _x,
                _y,
                _xInc,
                _yInc,
                _deviceTargetList.ptr,
                _deviceTargetList.size,
                _deviceTargetList.mask,
                _deviceResults,
                _deviceResultsCount);

        } else {
            _stepKernel->call(
                _blocks,
                _threads,
                _pointsPerThread,
                _compression,
                _chain,
                _x,
                _y,
                _xInc,
                _yInc,
                _deviceTargetList.ptr,
                _deviceTargetList.size,
                _deviceTargetList.mask,
                _deviceResults,
                _deviceResultsCount);
        }
        fflush(stdout);

        getResultsInternal();

        _iterations++;
    } catch(cl::CLException ex) {
        throw KeySearchException(ex.msg);
    }
}

void CLKeySearchDevice::setTargetsList()
{
    size_t count = _targetList.size();

    _targets = _clContext->malloc(5 * sizeof(unsigned int) * count);

    for(size_t i = 0; i < count; i++) {
        unsigned int h[5];

        undoRMD160FinalRound(_targetList[i].h, h);

        _clContext->copyHostToDevice(h, _targets, i * 5 * sizeof(unsigned int), 5 * sizeof(unsigned int));
    }

    _targetMemSize = count * 5 * sizeof(unsigned int);
    _deviceTargetList.ptr = _targets;
    _deviceTargetList.size = count;
}

void CLKeySearchDevice::setBloomFilter()
{
    uint64_t bloomFilterMask = getOptimalBloomFilterMask(1.0e-9, _targetList.size());

    initializeBloomFilter(_targetList, bloomFilterMask);
}

void CLKeySearchDevice::setTargetsInternal()
{
    // Clean up existing list
    if(_deviceTargetList.ptr != NULL) {
        _clContext->free(_deviceTargetList.ptr);
    }

    if(_targetList.size() < 16) {
        setTargetsList();
    } else {
        setBloomFilter();
    }
}

void CLKeySearchDevice::setTargets(const std::set<KeySearchTarget> &targets)
{
    try {
        _targetList.clear();

        for(std::set<KeySearchTarget>::iterator i = targets.begin(); i != targets.end(); ++i) {
            hash160 h(i->value);
            _targetList.push_back(h);
        }

        setTargetsInternal();
    } catch(cl::CLException ex) {
        throw KeySearchException(ex.msg);
    }
}

size_t CLKeySearchDevice::getResults(std::vector<KeySearchResult> &results)
{
    size_t count = _results.size();
    for(size_t i = 0; i < count; i++) {
        results.push_back(_results[i]);
    }
    _results.clear();

    return count;
}

uint32_t CLKeySearchDevice::keysPerIteration()
{
    return _threads * _blocks * _pointsPerThread;
}

std::string CLKeySearchDevice::getDeviceName()
{
    return _deviceName;
}

void CLKeySearchDevice::getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem)
{
    freeMem = _globalMemSize - _targetMemSize - _pointsMemSize;
    totalMem = _globalMemSize;
}

void CLKeySearchDevice::splatBigInt(secp256k1::uint256 &k, unsigned int *ptr)
{
    unsigned int buf[8];

    k.exportWords(buf, 8, secp256k1::uint256::BigEndian);

    memcpy(ptr, buf, sizeof(unsigned int) * 8);

}

bool CLKeySearchDevice::isTargetInList(const unsigned int hash[5])
{
    size_t count = _targetList.size();

    while(count) {
        if(memcmp(hash, _targetList[count - 1].h, 20) == 0) {
            return true;
        }
        count--;
    }

    return false;
}

void CLKeySearchDevice::removeTargetFromList(const unsigned int hash[5])
{
    size_t count = _targetList.size();

    while(count) {
        if(memcmp(hash, _targetList[count - 1].h, 20) == 0) {
            _targetList.erase(_targetList.begin() + count - 1);
            return;
        }
        count--;
    }
}

uint32_t CLKeySearchDevice::getPrivateKeyOffset(int thread, int block, int idx)
{
    // Total number of threads
    int totalThreads = _blocks * _threads;

    int base = idx * totalThreads;

    // Global ID of the current thread
    int threadId = block * _threads + thread;

    return base + threadId;
}

void CLKeySearchDevice::getResultsInternal()
{
    unsigned int numResults = 0;

    _clContext->copyDeviceToHost(_deviceResultsCount, &numResults, sizeof(unsigned int));

    if(numResults > 0) {
        CLDeviceResult *ptr = new CLDeviceResult[numResults];

        _clContext->copyDeviceToHost(_deviceResults, ptr, sizeof(CLDeviceResult) * numResults);

        unsigned int actualCount = 0;

        for(int i = 0; i < numResults; i++) {

            // might be false-positive
            if(!isTargetInList(ptr[i].digest)) {
                continue;
            }
            actualCount++;

            KeySearchResult minerResult;

            // Calculate the private key based on the number of iterations and the current thread
            uint64_t offset = (uint64_t)_blocks * _threads * _pointsPerThread * _iterations + getPrivateKeyOffset(ptr[i].thread, ptr[i].block, ptr[i].idx);
            secp256k1::uint256 privateKey = secp256k1::addModN(_start, secp256k1::uint256(offset));

            minerResult.privateKey = privateKey;
            minerResult.compressed = ptr[i].compressed;

            memcpy(minerResult.hash, ptr[i].digest, 20);

            minerResult.publicKey = secp256k1::ecpoint(secp256k1::uint256(ptr[i].x, secp256k1::uint256::BigEndian), secp256k1::uint256(ptr[i].y, secp256k1::uint256::BigEndian));

            removeTargetFromList(ptr[i].digest);

            _results.push_back(minerResult);
        }

        // Reset device counter
        numResults = 0;
        _clContext->copyHostToDevice(&numResults, _deviceResultsCount, sizeof(unsigned int));
    }
}

void CLKeySearchDevice::selfTest()
{
    unsigned int numPoints = _threads * _blocks * _pointsPerThread;
    std::vector<secp256k1::uint256> privateKeys;

    // Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
    secp256k1::uint256 privKey = _start;

    for(uint64_t i = 0; i < numPoints; i++) {
        privateKeys.push_back(privKey.add(i));
    }

    unsigned int *xBuf = new unsigned int[numPoints * 8];
    unsigned int *yBuf = new unsigned int[numPoints * 8];

    _clContext->copyDeviceToHost(_x, xBuf, sizeof(unsigned int) * 8 * numPoints);
    _clContext->copyDeviceToHost(_y, yBuf, sizeof(unsigned int) * 8 * numPoints);

    for(int block = 0; block < _blocks; block++) {
        for(int thread = 0; thread < _threads; thread++) {
            for(int idx = 0; idx < _pointsPerThread; idx++) {

                int index = getIndex(block, thread, idx);

                secp256k1::uint256 privateKey = privateKeys[index];

                secp256k1::uint256 x = readBigInt(xBuf, block, thread, idx);
                secp256k1::uint256 y = readBigInt(yBuf, block, thread, idx);

                secp256k1::ecpoint p1(x, y);
                secp256k1::ecpoint p2 = secp256k1::multiplyPoint(privateKey, secp256k1::G());

                if(!secp256k1::pointExists(p1)) {
                    throw std::string("Validation failed: invalid point");
                }

                if(!secp256k1::pointExists(p2)) {
                    throw std::string("Validation failed: invalid point");
                }

                if(!(p1 == p2)) {
                    throw std::string("Validation failed: points do not match");
                }
            }
        }
    }
}

void CLKeySearchDevice::splatBigInt(unsigned int *dest, int block, int thread, int idx, const secp256k1::uint256 &i)
{
    unsigned int value[8] = {0};

    i.exportWords(value, 8, secp256k1::uint256::BigEndian);

    int totalThreads = _blocks * _threads;
    int threadId = block * _threads + thread;

    int base = idx * totalThreads * 8;

    for(int k = 0; k < 8; k++) {
        dest[base + threadId * 8 + k] = value[k];
    }
}

secp256k1::uint256 CLKeySearchDevice::readBigInt(unsigned int *src, int block, int thread, int idx)
{
    unsigned int value[8] = {0};

    int totalThreads = _blocks * _threads;
    int threadId = block * _threads + thread;

    int base = idx * totalThreads * 8;

    for(int k = 0; k < 8; k++) {
        value[k] = src[base + threadId * 8 + k];
    }

    secp256k1::uint256 v(value, secp256k1::uint256::BigEndian);

    return v;
}

void CLKeySearchDevice::initializeBasePoints()
{
    // generate a table of points G, 2G, 4G, 8G...(2^255)G
    std::vector<secp256k1::ecpoint> table;

    table.push_back(secp256k1::G());
    for(int i = 1; i < 256; i++) {

        secp256k1::ecpoint p = doublePoint(table[i - 1]);
        if(!pointExists(p)) {
            throw std::string("Point does not exist!");
        }
        table.push_back(p);
    }

    size_t count = 256;

    unsigned int *tmpX = new unsigned int[count * 8];
    unsigned int *tmpY = new unsigned int[count * 8];

    for(int i = 0; i < 256; i++) {
        unsigned int bufX[8];
        unsigned int bufY[8];
        table[i].x.exportWords(bufX, 8, secp256k1::uint256::BigEndian);
        table[i].y.exportWords(bufY, 8, secp256k1::uint256::BigEndian);

        for(int j = 0; j < 8; j++) {
            tmpX[i * 8 + j] = bufX[j];
            tmpY[i * 8 + j] = bufY[j];
        }
    }

    _clContext->copyHostToDevice(tmpX, _xTable, count * 8 * sizeof(unsigned int));

    _clContext->copyHostToDevice(tmpY, _yTable, count * 8 * sizeof(unsigned int));
}

int CLKeySearchDevice::getIndex(int block, int thread, int idx)
{
    // Total number of threads
    int totalThreads = _blocks * _threads;

    int base = idx * totalThreads;

    // Global ID of the current thread
    int threadId = block * _threads + thread;

    return base + threadId;
}

void CLKeySearchDevice::generateStartingPoints()
{
    uint64_t totalPoints = _pointsPerThread * _threads * _blocks;
    uint64_t totalMemory = totalPoints * 40;

    std::vector<secp256k1::uint256> exponents;

    initializeBasePoints();

    _pointsMemSize = totalPoints * sizeof(unsigned int) * 16 + _pointsPerThread * sizeof(unsigned int) * 8;

    Logger::log(LogLevel::Info, "Generating " + util::formatThousands(totalPoints) + " starting points (" + util::format("%.1f", (double)totalMemory / (double)(1024 * 1024)) + "MB)");

    // Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
    secp256k1::uint256 privKey = _start;

    for(uint64_t i = 0; i < totalPoints; i++) {
        exponents.push_back(privKey.add(i));
    }

    unsigned int *privateKeys = new unsigned int[8 * totalPoints];

    for(int block = 0; block < _blocks; block++) {
        for(int thread = 0; thread < _threads; thread++) {
            for(int idx = 0; idx < _pointsPerThread; idx++) {

                int index = getIndex(block, thread, idx);

                splatBigInt(privateKeys, block, thread, idx, exponents[index]);
            }
        }
    }

    // Copy to device
    _clContext->copyHostToDevice(privateKeys, _privateKeys, totalPoints * 8 * sizeof(unsigned int));

    delete[] privateKeys;

    // Show progress in 10% increments
    double pct = 10.0;
    for(int i = 0; i < 256; i++) {
        _initKeysKernel->call(_blocks, _threads, _pointsPerThread, i, _privateKeys, _chain, _xTable, _yTable, _x, _y);

        if(((double)(i+1) / 256.0) * 100.0 >= pct) {
            Logger::log(LogLevel::Info, util::format("%.1f%%", pct));
            pct += 10.0;
        }
    }

    Logger::log(LogLevel::Info, "Done");
}