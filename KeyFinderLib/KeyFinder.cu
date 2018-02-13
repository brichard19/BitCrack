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


__device__ void setResultFound(unsigned int *numResultsPtr, void *results, int idx, bool compressed, unsigned int x[8], unsigned int y[8], unsigned int digest[5])
{
	grabLock();

	struct KeyFinderDeviceResult r;

	r.block = blockIdx.x;
	r.thread = threadIdx.x;
	r.idx = idx;
	r.compressed = compressed;

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

__device__ void doIteration(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chain, int pointsPerThread, unsigned int *numResults, void *results, int flags, int compression)
{
	// Multiply together all (_Gx - x) and then invert
	unsigned int inverse[8] = { 0,0,0,0,0,0,0,1 };
	for(int i = 0; i < pointsPerThread; i++) {
		unsigned int x[8];
		unsigned int y[8];
		unsigned int digest[5];

		readInt(xPtr, i, x);
		readInt(yPtr, i, y);

		if(compression == 1 || compression == 2) {
			hashPublicKey(x, y, digest);

			if(checkHash(digest)) {
				setResultFound(numResults, results, i, false, x, y, digest);
			}
		}

		if(compression == 0 || compression == 2) {
			hashPublicKeyCompressed(x, y, digest);

			if(checkHash(digest)) {
				setResultFound(numResults, results, i, true, x, y, digest);
			}
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

__device__ void doIterationWithDouble(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chain, int pointsPerThread, unsigned int *numResults, void *results, int flags, int compression)
{
	// Multiply together all (_Gx - x) and then invert
	unsigned int inverse[8] = { 0,0,0,0,0,0,0,1 };
	for(int i = 0; i < pointsPerThread; i++) {
		unsigned int x[8];
		unsigned int y[8];
		unsigned int digest[5];

		readInt(xPtr, i, x);
		readInt(yPtr, i, y);

		if(compression == 1 || compression == 2) {
			hashPublicKey(x, y, digest);

			if(checkHash(digest)) {
				setResultFound(numResults, results, i, false, x, y, digest);
			}
		}

		if(compression == 0 || compression == 2) {
			hashPublicKeyCompressed(x, y, digest);

			if(checkHash(digest)) {
				setResultFound(numResults, results, i, true, x, y, digest);
			}
		}

		beginBatchAddWithDouble(_INC_X, _INC_Y, xPtr, chain, i, inverse);
	}

	doBatchInverse(inverse);

	for(int i = pointsPerThread - 1; i >= 0; i--) {

		unsigned int newX[8];
		unsigned int newY[8];

		completeBatchAddWithDouble(_INC_X, _INC_Y, xPtr, yPtr, i, chain, inverse, newX, newY);

		writeInt(xPtr, i, newX);
		writeInt(yPtr, i, newY);
	}
}

/**
* Performs a single iteration
*/
__global__ void keyFinderKernel(int points, unsigned int flags, unsigned int *x, unsigned int *y, unsigned int *chain, unsigned int *numResults, void *results, int compression)
{
	doIteration(x, y, chain, points, numResults, results, flags, compression);
}

__global__ void keyFinderKernelWithDouble(int points, unsigned int flags, unsigned int *x, unsigned int *y, unsigned int *chain, unsigned int *numResults, void *results, int compression)
{
	doIterationWithDouble(x, y, chain, points, numResults, results, flags, compression);
}