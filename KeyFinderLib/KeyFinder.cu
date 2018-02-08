#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdlib.h>
#include <stdio.h>

#include "ptx.cuh"
#include "secp256k1.cuh"

#include "sha256.cuh"
#include "ripemd160.cuh"

#include "AddressMinerShared.h"
#include "secp256k1.h"


__constant__ unsigned int _TARGET_HASH[5];
__constant__ unsigned int _INC_X[8];
__constant__ unsigned int _INC_Y[8];


cudaError_t setTargetHash(const unsigned int hash[5])
{
	return cudaMemcpyToSymbol(_TARGET_HASH, hash, sizeof(unsigned int) * 5);
}

cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y)
{
	unsigned int xWords[8];
	unsigned int yWords[8];

	x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	cudaError_t err = cudaMemcpyToSymbol(_INC_X, xWords, sizeof(unsigned int) * 8);
	if(err) {
		return err;
	}

	return cudaMemcpyToSymbol(_INC_Y, yWords, sizeof(unsigned int) * 8);
}


#ifdef _DEBUG
__device__ bool verifyPoint(unsigned int *x, unsigned int *y)
{
	unsigned int y2[8];
	unsigned int x2[8];
	unsigned int x3[8];
	unsigned int seven[8] = { 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000007 };

	mulModP(y, y, y2);

	mulModP(x, x, x2);
	mulModP(x, x2, x3);

	unsigned int sum[8];
	addModP(x3, seven, sum);

	for(int i = 0; i < 8; i++) {
		if(y2[i] != sum[i]) {
			printf("y2':");
			printBigInt(y2, 8);
			printf("x3+7:");
			printBigInt(sum, 8);
			return false;
		}
	}

	return true;
}

__device__ bool checkInverse(const unsigned int *a, const unsigned int *b)
{
	unsigned int product[8] = { 0 };

	mulModP(a, b, product);

	for(int i = 0; i < 7; i++) {
		if(product[i] != 0) {
			return false;
		}
	}

	return product[7] == 1;
}
#endif

__device__ void hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digestOut)
{
	unsigned int hash[8];

	sha256PublicKey(x, y, hash);

	// Swap to little-endian
	for(int i = 0; i < 8; i++) {
		hash[i] = endian(hash[i]);
	}

	ripemd160sha256(hash, hash);

	for(int i = 0; i < 5; i++) {
		digestOut[i] = endian(hash[i]);
	}
}

__device__ void hashPublicKeyCompressed(const unsigned int *x, const unsigned int *y, unsigned int *digestOut)
{
	unsigned int hash[8];

	sha256PublicKeyCompressed(x, y, hash);

	// Swap to little-endian
	for(int i = 0; i < 8; i++) {
		hash[i] = endian(hash[i]);
	}

	ripemd160sha256(hash, hash);

	for(int i = 0; i < 5; i++) {
		digestOut[i] = endian(hash[i]);
	}
}

__device__ void setHashFoundFlag(unsigned int *flagsAra, int idx, int value)
{
	grabLock();
	int threadId = blockIdx.x * blockDim.x + threadIdx.x;

	int base = gridDim.x * blockDim.x * idx;

	flagsAra[base + threadId] = value;
	releaseLock();
}


__device__ void reportFoundHash(const unsigned int *x, const unsigned int *y, const unsigned int *digest)
{
#ifdef _DEBUG
	printf("============ FOUND HASH =============\n");
	printf("\nx:");
	printBigInt(x, 8);
	printf("y:");
	printBigInt(y, 8);
	printf("h:");
	printBigInt(digest, 5);
	printf("======================================\n");
#endif
}


__device__ void setResultFound(unsigned int *numResultsPtr, void *results, int idx, unsigned int x[8], unsigned int y[8], unsigned int digest[5])
{
	grabLock();

	struct KeyFinderDeviceResult r;

	r.block = blockIdx.x;
	r.thread = threadIdx.x;
	r.idx = idx;

	for(int i = 0; i < 8; i++) {
		r.x[i] = x[i];
		r.y[i] = y[i];
	}

	for(int i = 0; i < 5; i++) {
		r.digest[i] = digest[i];
	}

	struct KeyFinderDeviceResult *resultsPtr = (struct KeyFinderDeviceResult *)results;
	resultsPtr[*numResultsPtr] = r;
	(*numResultsPtr)++;
	releaseLock();
}

__device__ bool checkHash(unsigned int hash[5])
{
	bool equal = true;

	for(int i = 0; i < 5; i++) {
		equal &= (hash[i] == _TARGET_HASH[i]);
	}

	return equal;
}

__device__ void doIteration(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chain, int pointsPerThread, unsigned int *numResults, void *results, int flags)
{
	// Multiply together all (_Gx - x) and then invert
	unsigned int inverse[8] = { 0,0,0,0,0,0,0,1 };
	for(int i = 0; i < pointsPerThread; i++) {
		unsigned int x[8];
		unsigned int y[8];
		unsigned int digest[5];

		readInt(xPtr, i, x);
		readInt(yPtr, i, y);

		hashPublicKey(x, y, digest);

		if(checkHash(digest)) {
			setResultFound(numResults, results, i, x, y, digest);
		}

		beginBatchAdd(_INC_X, xPtr, chain, i, inverse);
	}

	doBatchInverse(inverse);

	for(int i = pointsPerThread - 1; i >= 0; i--) {

		unsigned int newX[8];
		unsigned int newY[8];

		completeBatchAdd(_INC_X, _INC_Y, xPtr, yPtr, i, chain, inverse, newX, newY);

		writeInt(xPtr, i, newX);
		writeInt(yPtr, i, newY);
	}
}

__device__ void doIterationWithDouble(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chain, int pointsPerThread, unsigned int *numResults, void *results, int flags)
{
	// Multiply together all (_Gx - x) and then invert
	unsigned int inverse[8] = { 0,0,0,0,0,0,0,1 };
	for(int i = 0; i < pointsPerThread; i++) {
		unsigned int x[8];
		unsigned int y[8];
		unsigned int digest[5];

		readInt(xPtr, i, x);
		readInt(yPtr, i, y);

		hashPublicKey(x, y, digest);

		if(checkHash(digest)) {
			setResultFound(numResults, results, i, x, y, digest);
		}

		//beginBatchAdd(_INC_X, xPtr, chain, i, inverse);
		beginBatchAddWithDouble(_INC_X, _INC_Y, xPtr, chain, i, inverse);
	}

	doBatchInverse(inverse);

	for(int i = pointsPerThread - 1; i >= 0; i--) {

		unsigned int newX[8];
		unsigned int newY[8];

		//completeBatchAdd(_INC_X, _INC_Y, xPtr, yPtr, i, chain, inverse, newX, newY);
		completeBatchAddWithDouble(_INC_X, _INC_Y, xPtr, yPtr, i, chain, inverse, newX, newY);

		writeInt(xPtr, i, newX);
		writeInt(yPtr, i, newY);
	}
}

/**
* Performs a single iteration
*/
__global__ void keyFinderKernel(int points, unsigned int flags, unsigned int *x, unsigned int *y, unsigned int *chain, unsigned int *numResults, void *results, bool useDouble)
{
	//if(useDouble) {
	//	doIterationWithDouble(x, y, chain, points, numResults, results, flags);
	//} else {
	//	doIteration(x, y, chain, points, numResults, results, flags);
	//}
	doIteration(x, y, chain, points, numResults, results, flags);
}