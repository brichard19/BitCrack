#include "sha256.cl"
#include "ripemd160.cl"
#include "secp256k1.cl"

#define COMPRESSED 0
#define UNCOMPRESSED 1
#define BOTH 2

/*
typedef struct {
    ulong mask;
    ulong size;
    unsigned int *ptr;
}CLTargetList;
*/

unsigned int endian(unsigned int x)
{
    return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

typedef struct {
    int thread;
    int block;
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
}CLDeviceResult;

bool isInList(unsigned int hash[5], __global unsigned int *targetList, size_t numTargets)
{
    int found = 0;

    for(size_t i = 0; i < numTargets; i++) {
        int equal = 0;

        for(int j = 0; j < 5; j++) {
            if(hash[j] == targetList[5 * i + j]) {
                equal++;
            }
        }

        if(equal == 5) {
            found = 1;
        }
    }

    return found;
}

bool isInBloomFilter(unsigned int hash[5], __global unsigned int *targetList, ulong mask)
{
    bool foundMatch = true;

    unsigned int h5 = 0;
    /*
    if(get_local_id(0) == 0) {
        for(int i = 0; i < 5; i++) {
            printf("%.8x ", hash[i]);
        }
        printf("\n");
    }
    */
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


void printInt(unsigned int *x)
{
    if(get_local_id(0) == 0) {
        for(int i = 0; i < 8; i++) {
            printf("%.8x ", x[i]);
        }
        printf("\n");
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
    int pointsPerThread,
    int step,
    __global unsigned int *privateKeys,
    __global unsigned int *chain,
    __global unsigned int *gxPtr,
    __global unsigned int *gyPtr,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr)
{
    unsigned int gx[8];
    unsigned int gy[8];

    for(int i = 0; i < 8; i++) {
        gx[i] = gxPtr[step * 8 + i];
        gy[i] = gyPtr[step * 8 + i];
    }

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};

    int batchIdx = 0;
    for(int i = 0; i < pointsPerThread; i++) {

        unsigned int p[8];
        readInt(privateKeys, i, p);

        unsigned int bit = p[7 - step / 32] & (1 << (step % 32));


        unsigned int x[8];
        readInt(xPtr, i, x);
        

        if(bit != 0) {
            if(!isInfinity(x)) {
                beginBatchAddWithDouble(gx, gy, xPtr, chain, i, batchIdx, inverse);
                batchIdx++;
            }
        }
    }

    doBatchInverse(inverse);


    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        unsigned int p[8];
        readInt(privateKeys, i, p);
        unsigned int bit = p[7 - step / 32] & (1 << (step % 32));

        unsigned int x[8];
        readInt(xPtr, i, x);

        bool infinity = isInfinity(x);

        if(bit != 0) {
            if(!infinity) {
                batchIdx--;
                completeBatchAddWithDouble(gx, gy, xPtr, yPtr, i, batchIdx, chain, inverse, newX, newY);
            } else {
                copyBigInt(gx, newX);
                copyBigInt(gy, newY);
            }

            writeInt(xPtr, i, newX);
            writeInt(yPtr, i, newY);

        }
    }
}


void hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKey(x, y, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

void hashPublicKeyCompressed(const unsigned int *x, unsigned int yParity, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x, yParity, hash);

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

void setResultFound(int idx, bool compressed, unsigned int x[8], unsigned int y[8], unsigned int digest[5], __global CLDeviceResult *results, __global unsigned int *numResults)
{
    CLDeviceResult r;

    r.block = get_group_id(0);
    r.thread = get_local_id(0);
    r.idx = idx;
    r.compressed = compressed;

    for(int i = 0; i < 8; i++) {
        r.x[i] = x[i];
        r.y[i] = y[i];
    }

    doRMD160FinalRound(digest, r.digest);

    atomicListAdd(results, numResults, &r);
}

void doIteration(
    size_t pointsPerThread,
    int compression,
    __global unsigned int *chain,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr,
    __global unsigned int *incXPtr,
    __global unsigned int *incYPtr,
    __global unsigned int *targetList,
    size_t numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    unsigned int incX[8];
    unsigned int incY[8];

    for(int i = 0; i < 8; i++) {
        incX[i] = incXPtr[i];
        incY[i] = incYPtr[i];
    }

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];

        unsigned int digest[5];

        readInt(xPtr, i, x);

        if((compression == UNCOMPRESSED) || (compression == BOTH)) {
            unsigned int y[8];
            readInt(yPtr, i, y);

            hashPublicKey(x, y, digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW(yPtr, i), digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
                unsigned int y[8];
                readInt(yPtr, i, y);
                setResultFound(i, true, x, y, digest, results, numResults);
            }
        }

        beginBatchAdd(incX, x, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAdd(incX, incY, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}


void doIterationWithDouble(
    size_t pointsPerThread,
    int compression,
    __global unsigned int *chain,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr,
    __global unsigned int *incXPtr,
    __global unsigned int *incYPtr,
    __global unsigned int *targetList,
    size_t numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    unsigned int incX[8];
    unsigned int incY[8];

    for(int i = 0; i < 8; i++) {
        incX[i] = incXPtr[i];
        incY[i] = incYPtr[i];
    }

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];

        unsigned int digest[5];

        readInt(xPtr, i, x);

        // uncompressed
        if((compression == UNCOMPRESSED) || (compression == BOTH)) {
            unsigned int y[8];
            readInt(yPtr, i, y);
            hashPublicKey(x, y, digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        // compressed
        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW(yPtr, i), digest);

            if(checkHash(digest, targetList, numTargets, mask)) {

                unsigned int y[8];
                readInt(yPtr, i, y);

                setResultFound(i, true, x, y, digest, results, numResults);
            }
        }

        beginBatchAddWithDouble(incX, incY, xPtr, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAddWithDouble(incX, incY, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}

/**
* Performs a single iteration
*/
__kernel void keyFinderKernel(
    unsigned int pointsPerThread,
    int compression,
    __global unsigned int *chain,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr,
    __global unsigned int *incXPtr,
    __global unsigned int *incYPtr,
    __global unsigned int *targetList,
    ulong numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    doIteration(pointsPerThread, compression, chain, xPtr, yPtr, incXPtr, incYPtr, targetList, numTargets, mask, results, numResults);
}

__kernel void keyFinderKernelWithDouble(
    unsigned int pointsPerThread,
    int compression,
    __global unsigned int *chain,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr,
    __global unsigned int *incXPtr,
    __global unsigned int *incYPtr,
    __global unsigned int *targetList,
    ulong numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    doIterationWithDouble(pointsPerThread, compression, chain, xPtr, yPtr, incXPtr, incYPtr, targetList, numTargets, mask, results, numResults);
}