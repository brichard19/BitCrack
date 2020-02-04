#define COMPRESSED 0
#define UNCOMPRESSED 1
#define BOTH 2

unsigned int endian(unsigned int x)
{
    return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

typedef struct {
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
}CLDeviceResult;

bool isInList(unsigned int hash[5], __global unsigned int *targetList, size_t numTargets)
{
    bool found = false;

    for(size_t i = 0; i < numTargets; i++) {
        int equal = 0;

        for(int j = 0; j < 5; j++) {
            if(hash[j] == targetList[5 * i + j]) {
                equal++;
            }
        }

        if(equal == 5) {
            found = true;
        }
    }

    return found;
}

bool isInBloomFilter(unsigned int hash[5], __global unsigned int *targetList, ulong mask)
{
    bool foundMatch = true;

    unsigned int h5 = 0;

    for(int i = 0; i < 5; i++) {
        h5 += hash[i];
    }

    uint64_t idx[5];

    idx[0] = ((hash[0] << 6) | (h5 & 0x3f)) & mask;
    idx[1] = ((hash[1] << 6) | ((h5 >> 6) & 0x3f)) & mask;
    idx[2] = ((hash[2] << 6) | ((h5 >> 12) & 0x3f)) & mask;
    idx[3] = ((hash[3] << 6) | ((h5 >> 18) & 0x3f)) & mask;
    idx[4] = ((hash[4] << 6) | ((h5 >> 24) & 0x3f)) & mask;

    for(int i = 0; i < 5; i++) {
        unsigned int j = idx[i];
        unsigned int f = targetList[j / 32];

        if((f & (0x01 << (j % 32))) == 0) {
            foundMatch = false;
        }
    }

    return foundMatch;
}

bool checkHash(unsigned int hash[5], __global unsigned int *targetList, size_t numTargets, ulong mask)
{
    if(numTargets > 16) {
        return isInBloomFilter(hash, targetList, mask);
    } else {
        return isInList(hash, targetList, numTargets);
    }
}


void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
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


__kernel void multiplyStepKernel(
    int totalPoints,
    int step,
    __global uint256_t* privateKeys,
    __global uint256_t* chain,
    __global uint256_t* gxPtr,
    __global uint256_t* gyPtr,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr)
{
    uint256_t gx;
    uint256_t gy;
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    gx = gxPtr[step];
    gy = gyPtr[step];

    // Multiply together all (_Gx - x) and then invert
    uint256_t inverse = { {0,0,0,0,0,0,0,1} };

    int batchIdx = 0;
    int i = gid;
    for(; i < totalPoints; i += dim) {

        unsigned int p;
        p = readWord256k(privateKeys, i, 7 - step / 32);

        unsigned int bit = p & (1 << (step % 32));

        uint256_t x = xPtr[i];

        if(bit != 0) {
            if(!isInfinity256k(x)) {
                beginBatchAddWithDouble256k(gx, gy, xPtr, chain, i, batchIdx, &inverse);
                batchIdx++;
            }
        }
    }

    //doBatchInverse(inverse);
    inverse = doBatchInverse256k(inverse);

    i -= dim;
    for(; i >= 0; i -= dim) {
        uint256_t newX;
        uint256_t newY;

        unsigned int p;
        p = readWord256k(privateKeys, i, 7 - step / 32);
        unsigned int bit = p & (1 << (step % 32));

        uint256_t x = xPtr[i];
        bool infinity = isInfinity256k(x);

        if(bit != 0) {
            if(!infinity) {
                batchIdx--;
                completeBatchAddWithDouble256k(gx, gy, xPtr, yPtr, i, batchIdx, chain, &inverse, &newX, &newY);
            } else {
                newX = gx;
                newY = gy;
            }

            xPtr[i] = newX;
            yPtr[i] = newY;
        }
    }
}


void hashPublicKey(uint256_t x, uint256_t y, unsigned int* digestOut)
{
    unsigned int hash[8];

    sha256PublicKey(x.v, y.v, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

void hashPublicKeyCompressed(uint256_t x, unsigned int yParity, unsigned int* digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x.v, yParity, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);

}

void atomicListAdd(__global CLDeviceResult *results, __global unsigned int *numResults, CLDeviceResult *r)
{
    unsigned int count = atomic_add(numResults, 1);

    results[count] = *r;
}

void setResultFound(int idx, bool compressed, uint256_t x, uint256_t y, unsigned int digest[5], __global CLDeviceResult* results, __global unsigned int* numResults)
{
    CLDeviceResult r;

    r.idx = idx;
    r.compressed = compressed;

    for(int i = 0; i < 8; i++) {
        r.x[i] = x.v[i];
        r.y[i] = y.v[i];
    }

    doRMD160FinalRound(digest, r.digest);

    atomicListAdd(results, numResults, &r);
}

void doIteration(
    size_t totalPoints,
    int compression,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int *targetList,
    size_t numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    uint256_t incX = *incXPtr;
    uint256_t incY = *incYPtr;

    // Multiply together all (_Gx - x) and then invert
    uint256_t inverse = { {0,0,0,0,0,0,0,1} };
    int i = gid;
    int batchIdx = 0;

    for(; i < totalPoints; i += dim) {
        uint256_t x;

        unsigned int digest[5];

        x = xPtr[i];

        if((compression == UNCOMPRESSED) || (compression == BOTH)) {
            uint256_t y = yPtr[i];

            hashPublicKey(x, y, digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW256k(yPtr, i), digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
                uint256_t y = yPtr[i];
                setResultFound(i, true, x, y, digest, results, numResults);
            }
        }

        beginBatchAdd256k(incX, x, chain, i, batchIdx, &inverse);
        batchIdx++;
    }

    inverse = doBatchInverse256k(inverse);

    i -= dim;

    for(;  i >= 0; i -= dim) {

        uint256_t newX;
        uint256_t newY;
        batchIdx--;
        completeBatchAdd256k(incX, incY, xPtr, yPtr, i, batchIdx, chain, &inverse, &newX, &newY);

        xPtr[i] = newX;
        yPtr[i] = newY;
    }
}


void doIterationWithDouble(
    size_t totalPoints,
    int compression,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int* targetList,
    size_t numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    uint256_t incX = *incXPtr;
    uint256_t incY = *incYPtr;

    // Multiply together all (_Gx - x) and then invert
    uint256_t inverse = { {0,0,0,0,0,0,0,1} };

    int i = gid;
    int batchIdx = 0;
    for(; i < totalPoints; i += dim) {
        uint256_t x;

        unsigned int digest[5];

        x = xPtr[i];

        // uncompressed
        if((compression == UNCOMPRESSED) || (compression == BOTH)) {
            uint256_t y = yPtr[i];
            hashPublicKey(x, y, digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        // compressed
        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW256k(yPtr, i), digest);

            if(checkHash(digest, targetList, numTargets, mask)) {

                uint256_t y = yPtr[i];
                setResultFound(i, true, x, y, digest, results, numResults);
            }
        }

        beginBatchAddWithDouble256k(incX, incY, xPtr, chain, i, batchIdx, &inverse);
        batchIdx++;
    }

    inverse = doBatchInverse256k(inverse);

    i -= dim;

    for(; i >= 0; i -= dim) {
        uint256_t newX;
        uint256_t newY;
        batchIdx--;
        completeBatchAddWithDouble256k(incX, incY, xPtr, yPtr, i, batchIdx, chain, &inverse, &newX, &newY);

        xPtr[i] = newX;
        yPtr[i] = newY;
    }
}

/**
* Performs a single iteration
*/
__kernel void keyFinderKernel(
    unsigned int totalPoints,
    int compression,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int* targetList,
    ulong numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    doIteration(totalPoints, compression, chain, xPtr, yPtr, incXPtr, incYPtr, targetList, numTargets, mask, results, numResults);
}

__kernel void keyFinderKernelWithDouble(
    unsigned int totalPoints,
    int compression,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int* targetList,
    ulong numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    doIterationWithDouble(totalPoints, compression, chain, xPtr, yPtr, incXPtr, incYPtr, targetList, numTargets, mask, results, numResults);
}
