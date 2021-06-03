#define COMPRESSED 0
#define UNCOMPRESSED 1
#define BOTH 2

#ifndef endian
#define endian(x) ((x) << 24) | (((x) << 8) & 0x00ff0000) | (((x) >> 8) & 0x0000ff00) | ((x) >> 24)
#endif

typedef struct {
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
}CLDeviceResult;

bool isInBloomFilter(unsigned int hash[5], __global unsigned int *targetList, ulong *mask)
{
    unsigned int h5 = hash[0] + hash[1] + hash[2] + hash[3] + hash[4];

    return (false == 
        (
            (targetList[(((hash[0] << 6) | (h5 & 0x3f)) & *mask) / 32] & (0x01 << ((((hash[0] << 6) | (h5 & 0x3f)) & *mask) % 32))) == 0 ||
            (targetList[(((hash[1] << 6) | ((h5 >> 6) & 0x3f)) & *mask) / 32] & (0x01 << ((((hash[1] << 6) | ((h5 >> 6) & 0x3f)) & *mask) % 32))) == 0 ||
            (targetList[(((hash[2] << 6) | ((h5 >> 12) & 0x3f)) & *mask) / 32] & (0x01 << ((((hash[2] << 6) | ((h5 >> 12) & 0x3f)) & *mask) % 32))) == 0 ||
            (targetList[(((hash[3] << 6) | ((h5 >> 18) & 0x3f)) & *mask) / 32] & (0x01 << ((((hash[3] << 6) | ((h5 >> 18) & 0x3f)) & *mask) % 32))) == 0 || 
            (targetList[ (((hash[4] << 6) | ((h5 >> 24) & 0x3f)) & *mask) / 32] & (0x01 << ( (((hash[4] << 6) | ((h5 >> 24) & 0x3f)) & *mask) % 32))) == 0
        )
    );
}

inline void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    hOut[0] = endian(hIn[0] + 0xefcdab89);
    hOut[1] = endian(hIn[1] + 0x98badcfe);
    hOut[2] = endian(hIn[2] + 0x10325476);
    hOut[3] = endian(hIn[3] + 0xc3d2e1f0);
    hOut[4] = endian(hIn[4] + 0x67452301);
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
    int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    gx = gxPtr[step];
    gy = gyPtr[step];

    // Multiply together all (_Gx - x) and then invert
    uint256_t inverse = { {0,0,0,0,0,0,0,1} };

    int batchIdx = 0;
    uint256_t x;

    for(; i < totalPoints; i += dim) {
        if(( (readWord256k(privateKeys, i, 7 - step / 32)) & (1 << (step % 32))) != 0) {
            x = xPtr[i];
            if(!isInfinity256k(x.v)) {
                beginBatchAddWithDouble256k(gx, gy, xPtr, chain, i, batchIdx, &inverse);
                batchIdx++;
            }
        }
    }

    doBatchInverse256k(inverse.v);

    uint256_t newX;
    uint256_t newY;
    i -= dim;
    for(; i >= 0; i -= dim) {
        x = xPtr[i];

        if(((readWord256k(privateKeys, i, 7 - step / 32)) & (1 << (step % 32))) != 0) {
            if(!isInfinity256k(x.v)) {
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


void hashPublicKey(uint256_t x, uint256_t y, unsigned int digest[5])
{
    unsigned int hash[8];

    sha256PublicKey(x.v, y.v, hash);

    // Swap to little-endian
    hash[0] = endian(hash[0]);
    hash[1] = endian(hash[1]);
    hash[2] = endian(hash[2]);
    hash[3] = endian(hash[3]);
    hash[4] = endian(hash[4]);
    hash[5] = endian(hash[5]);
    hash[6] = endian(hash[6]);
    hash[7] = endian(hash[7]);

    ripemd160sha256NoFinal(hash, digest);
}

void hashPublicKeyCompressed(uint256_t x, unsigned int yParity, unsigned int digest[5])
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x.v, yParity, hash);

    // Swap to little-endian
    hash[0] = endian(hash[0]);
    hash[1] = endian(hash[1]);
    hash[2] = endian(hash[2]);
    hash[3] = endian(hash[3]);
    hash[4] = endian(hash[4]);
    hash[5] = endian(hash[5]);
    hash[6] = endian(hash[6]);
    hash[7] = endian(hash[7]);

    ripemd160sha256NoFinal(hash, digest);
}

void setResultFound(int idx, bool compressed, uint256_t x, uint256_t y, unsigned int digest[5], __global CLDeviceResult* results, __global unsigned int* numResults)
{
    CLDeviceResult r;

    r.idx = idx;
    r.compressed = compressed;

    r.x[0] = x.v[0];
    r.y[0] = y.v[0];

    r.x[1] = x.v[1];
    r.y[1] = y.v[1];
    
    r.x[2] = x.v[2];
    r.y[2] = y.v[2];
    
    r.x[3] = x.v[3];
    r.y[3] = y.v[3];

    r.x[4] = x.v[4];
    r.y[4] = y.v[4];
    
    r.x[5] = x.v[5];
    r.y[5] = y.v[5];
    
    r.x[6] = x.v[6];
    r.y[6] = y.v[6];
    
    r.x[7] = x.v[7];
    r.y[7] = y.v[7];

    doRMD160FinalRound(digest, r.digest);

    results[atomic_add(numResults, 1)] = r;
}

__kernel void keyFinderKernel(
    unsigned int totalPoints,
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
    int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    uint256_t incX = *incXPtr;
    uint256_t incY = *incYPtr;

    // Multiply together all (_Gx - x) and then invert
    uint256_t inverse = { {0,0,0,0,0,0,0,1} };
    int batchIdx = 0;

    unsigned int digest[5];

#ifdef COMPRESSION_UNCOMPRESSED
    for(; i < totalPoints; i += dim) {
        hashPublicKey(xPtr[i], yPtr[i], digest);

        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, false, xPtr[i], yPtr[i], digest, results, numResults);
        }

        beginBatchAdd256k(incX, xPtr[i], chain, i, batchIdx, &inverse);
        batchIdx++;
    }
#elif COMPRESSION_BOTH
    for(; i < totalPoints; i += dim) {
        hashPublicKey(xPtr[i], yPtr[i], digest);

        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, false, xPtr[i], yPtr[i], digest, results, numResults);
        }

        hashPublicKeyCompressed(xPtr[i], readLSW256k(yPtr, i), digest);

        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, true, xPtr[i], yPtr[i], digest, results, numResults);
        }

        beginBatchAdd256k(incX, xPtr[i], chain, i, batchIdx, &inverse);
        batchIdx++;
    }
#else
    for(; i < totalPoints; i += dim) {
        hashPublicKeyCompressed(xPtr[i], readLSW256k(yPtr, i), digest);

        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, true, xPtr[i], yPtr[i], digest, results, numResults);
        }

        beginBatchAdd256k(incX, xPtr[i], chain, i, batchIdx, &inverse);
        batchIdx++;
    }
#endif

    doBatchInverse256k(inverse.v);

    i -= dim;
    uint256_t newX;
    uint256_t newY;
    for(;  i >= 0; i -= dim) {

        batchIdx--;
        completeBatchAdd256k(incX, incY, xPtr, yPtr, i, batchIdx, chain, &inverse, &newX, &newY);

        xPtr[i] = newX;
        yPtr[i] = newY;
    }
}

__kernel void keyFinderKernelWithDouble(
    unsigned int totalPoints,
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
    int i = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    uint256_t incX = *incXPtr;
    uint256_t incY = *incYPtr;

    // Multiply together all (_Gx - x) and then invert
    uint256_t inverse = { {0,0,0,0,0,0,0,1} };

    int batchIdx = 0;
    unsigned int digest[5];

#ifdef COMPRESSION_UNCOMPRESSED
    for(; i < totalPoints; i += dim) {
        hashPublicKey(xPtr[i], yPtr[i], digest);

        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, false, xPtr[i], yPtr[i], digest, results, numResults);
        }

        beginBatchAddWithDouble256k(incX, incY, xPtr, chain, i, batchIdx, &inverse);
        batchIdx++;
    }
#elif COMPRESSION_BOTH
    for(; i < totalPoints; i += dim) {
        hashPublicKey(xPtr[i], yPtr[i], digest);

        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, false, xPtr[i], yPtr[i], digest, results, numResults);
        }

        hashPublicKeyCompressed(xPtr[i], readLSW256k(yPtr, i), digest);

        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, true, xPtr[i], yPtr[i], digest, results, numResults);
        }

        beginBatchAddWithDouble256k(incX, incY, xPtr, chain, i, batchIdx, &inverse);
        batchIdx++;
    }
#else
    for(; i < totalPoints; i += dim) {
        hashPublicKeyCompressed(xPtr[i], readLSW256k(yPtr, i), digest);
        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, true, xPtr[i], yPtr[i], digest, results, numResults);
        }

        beginBatchAddWithDouble256k(incX, incY, xPtr, chain, i, batchIdx, &inverse);
        batchIdx++;
    }
#endif

    doBatchInverse256k(inverse.v);

    i -= dim;

    uint256_t newX;
    uint256_t newY;
    for(; i >= 0; i -= dim) {
        batchIdx--;
        completeBatchAddWithDouble256k(incX, incY, xPtr, yPtr, i, batchIdx, chain, &inverse, &newX, &newY);

        xPtr[i] = newX;
        yPtr[i] = newY;
    }
}
