#define COMPRESSED 0
#define UNCOMPRESSED 1
#define BOTH 2

typedef struct {
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
}CLDeviceResult;

void setResultFound(
    const int idx,
    const bool compressed,
    const uint256_t x,
    const uint256_t y,
    const unsigned int digest[5],
    __global CLDeviceResult* results,
    __global unsigned int* numResults
) {
    CLDeviceResult r;

    r.idx = idx;
    r.compressed = compressed;

    r.x[0] = x.v[0];
    r.x[1] = x.v[1];
    r.x[2] = x.v[2];
    r.x[3] = x.v[3];
    r.x[4] = x.v[4];
    r.x[5] = x.v[5];
    r.x[6] = x.v[6];
    r.x[7] = x.v[7];

    r.y[0] = y.v[0];
    r.y[1] = y.v[1];
    r.y[2] = y.v[2];
    r.y[3] = y.v[3];
    r.y[4] = y.v[4];
    r.y[5] = y.v[5];
    r.y[6] = y.v[6];
    r.y[7] = y.v[7];

    ripemd160FinalRound(digest, r.digest);

    results[atomic_add(numResults, 1)] = r;
}

__kernel void _initKeysKernel(
    const unsigned int totalPoints,
    const unsigned int step,
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

__kernel void _stepKernel(
    const unsigned int totalPoints,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int* targetList,
    const ulong mask,
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

    for(; i < totalPoints; i += dim) {
       
#if defined(COMPRESSION_UNCOMPRESSED) || defined(COMPRESSION_BOTH)
        hashPublicKey(xPtr[i], yPtr[i], digest);

        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, false, xPtr[i], yPtr[i], digest, results, numResults);
        }
#endif
#if defined(COMPRESSION_COMPRESSED) || defined(COMPRESSION_BOTH)
        hashPublicKeyCompressed(xPtr[i], readLSW256k(yPtr, i), digest);

        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, true, xPtr[i], yPtr[i], digest, results, numResults);
        }
#endif
        beginBatchAdd256k(incX, xPtr[i], chain, i, batchIdx, &inverse);
        batchIdx++;
    }

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

__kernel void _stepKernelWithDouble(
    const unsigned int totalPoints,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int* targetList,
    const ulong mask,
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

    for(; i < totalPoints; i += dim) {
#if defined(COMPRESSION_UNCOMPRESSED) || defined(COMPRESSION_BOTH)
        hashPublicKey(xPtr[i], yPtr[i], digest);
        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, false, xPtr[i], yPtr[i], digest, results, numResults);
        }
#endif
#if defined(COMPRESSION_COMPRESSED) || defined(COMPRESSION_BOTH)
        hashPublicKeyCompressed(xPtr[i], readLSW256k(yPtr, i), digest);
        if(isInBloomFilter(digest, targetList, &mask)) {
            setResultFound(i, true, xPtr[i], yPtr[i], digest, results, numResults);
        }
#endif

        beginBatchAddWithDouble256k(incX, incY, xPtr, chain, i, batchIdx, &inverse);
        batchIdx++;
    }

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
