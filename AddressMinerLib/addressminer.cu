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


__constant__ unsigned int _TARGET_MIN[5];
__constant__ unsigned int _TARGET_MAX[5];

cudaError_t setMinMaxTarget(const unsigned int *min, const unsigned int *max, const unsigned int *qx, const unsigned int *qy)
{
	cudaError_t err;

	err = cudaMemcpyToSymbol(_TARGET_MIN, min, sizeof(unsigned int) * 5);
	if(err != cudaSuccess) {
		return err;
	}

	err = cudaMemcpyToSymbol(_TARGET_MAX, max, sizeof(unsigned int) * 5);
	if(err != cudaSuccess) {
		return err;
	}

	err = cudaMemcpyToSymbol(_QX, qx, sizeof(unsigned int) * 8);
	if(err != cudaSuccess) {
		return err;
	}

	err = cudaMemcpyToSymbol(_QY, qy, sizeof(unsigned int) * 8);
	if(err != cudaSuccess) {
		return err;
	}

	return err;
}


#ifdef _DEBUG
__device__ bool verifyPoint(unsigned int *x, unsigned int *y)
{
	unsigned int y2[8];
	unsigned int x2[8];
	unsigned int x3[8];
	unsigned int seven[8] = {0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000000, 0x00000007};

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

__device__ __forceinline__ bool checkHash(const unsigned int *hash)
{
	if(hash[0] > _TARGET_MAX[0] || hash[0] < _TARGET_MIN[0]) {
		return false;
	}

	for(int i = 0; i < 5; i++) {
		if(hash[i] == _TARGET_MIN[i] || hash[i] == _TARGET_MAX[i]) {
			continue;
		} else if(hash[i] > _TARGET_MIN[i] && hash[i] < _TARGET_MAX[i]) {
			return true;
		} else {
			return false;
		}
	}

	return true;
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


__device__ void setResultFound(unsigned int *numResultsPtr, void *results, int autoType, bool compressed, int idx, unsigned int x[8], unsigned int y[8])
{
	grabLock();

	struct AddressMinerDeviceResult r;

	r.automorphism = autoType;
	r.block = blockIdx.x;
	r.thread = threadIdx.x;
	r.idx = idx;
	r.compressed = compressed;

	for(int i = 0; i < 8; i++) {
		r.x[i] = x[i];
		r.y[i] = y[i];
	}

	struct AddressMinerDeviceResult *resultsPtr = (struct AddressMinerDeviceResult *)results;
	resultsPtr[*numResultsPtr] = r;
	(*numResultsPtr)++;
	releaseLock();
}

__device__ void doIteration(unsigned int *xPtr, unsigned int *yPtr, unsigned int *chain, int pointsPerThread, unsigned int *numResults, void *results, int flags)
{
	// Multiply together all (_Gx - x) and then invert
	unsigned int inverse[8] = { 0,0,0,0,0,0,0,1 };
	for(int i = 0; i < pointsPerThread; i++) {
		beginBatchAdd(xPtr, chain, i, inverse);
	}

	doBatchInverse(inverse);

	for(int i = pointsPerThread - 1; i >= 0; i--) {

		unsigned int newX[8];
		unsigned int newY[8];

		completeBatchAdd(xPtr, yPtr, i, chain, inverse, newX, newY);

		writeInt(xPtr, i, newX);
		writeInt(yPtr, i, newY);

		unsigned int digest[5];

		if(flags & PointCompressionType::UNCOMPRESSED) {
			// Hash uncompressed
			hashPublicKey(newX, newY, digest);

			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::NONE, false, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}

		if(flags & PointCompressionType::COMPRESSED) {
			// Hash compressed
			hashPublicKeyCompressed(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::NONE, true, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}


		// Compute negative point
		negModP(newY, newY);

		if(flags & PointCompressionType::UNCOMPRESSED) {
			// Hash the public key
			hashPublicKey(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::NEGATIVE, false, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}
		
		if(flags & PointCompressionType::COMPRESSED) {
			// Hash compressed
			hashPublicKeyCompressed(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::NEGATIVE, true, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}


		// Apply automorphism 1 (negative)
		mulModP(_BETA, newX);

		if(flags & PointCompressionType::UNCOMPRESSED) {
			// Hash uncompressed
			hashPublicKey(newX, newY, digest);
			if(checkHash(digest)) {
				//*hashFoundGlobalFlag = 1;
				//setHashFoundFlag(hashFoundFlags, i, FOUND_NEG_AUTO1);
				setResultFound(numResults, results, AutomorphismType::TYPE1_NEGATIVE, false, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}

		if(flags & PointCompressionType::COMPRESSED) {
			// Hash compressed
			hashPublicKeyCompressed(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::TYPE1_NEGATIVE, true, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}

		// Negate point
		negModP(newY, newY);

		// Automorphism 1 (positive)

		if(flags & PointCompressionType::UNCOMPRESSED) {
			// Hash uncompressed
			hashPublicKey(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::TYPE1, false, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}

		if(flags & PointCompressionType::COMPRESSED) {
			// Hash compressed
			hashPublicKeyCompressed(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::TYPE1, true, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}

		// Apply automorphism 2 (positive)
		mulModP(_BETA, newX);

		if(flags & PointCompressionType::UNCOMPRESSED) {
			// Hash uncompressed
			hashPublicKey(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::TYPE2, false, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}

		if(flags & PointCompressionType::COMPRESSED) {
			// Hash compressed
			hashPublicKeyCompressed(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::TYPE2, true, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}

		// Negate point
		negModP(newY, newY);

		if(flags & PointCompressionType::UNCOMPRESSED) {
			// Automorphism 2 (negative)
			// Hash uncompressed
			hashPublicKey(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::TYPE2_NEGATIVE, false, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}

		if(flags & PointCompressionType::COMPRESSED) {
			// Hash compressed
			hashPublicKeyCompressed(newX, newY, digest);
			if(checkHash(digest)) {
				setResultFound(numResults, results, AutomorphismType::TYPE2_NEGATIVE, true, i, newX, newY);
				reportFoundHash(newX, newY, digest);
			}
		}
	}
}


/**
 * Performs a single iteration
 */
__global__ void addressMinerKernel(int points, unsigned int flags, unsigned int *x, unsigned int *y, unsigned int *chain, unsigned int *numResults, void *results)
{
	doIteration(x, y, chain, points, numResults, results, flags);
}