#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <vector>

#include "KeySearchDevice.h"

#include "CudaHashLookup.h"

#include "CudaHashLookup.cuh"

#include "Logger.h"

#include "util.h"

#define MAX_TARGETS_CONSTANT_MEM 16

__constant__ unsigned int _TARGET_HASH[MAX_TARGETS_CONSTANT_MEM][5];
__constant__ unsigned int _NUM_TARGET_HASHES[1];
__constant__ unsigned int *_BLOOM_FILTER[1];
__constant__ unsigned int _BLOOM_FILTER_MASK[1];
__constant__ unsigned long long _BLOOM_FILTER_MASK64[1];

__constant__ unsigned int _USE_BLOOM_FILTER[1];


static unsigned int swp(unsigned int x)
{
	return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

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
		hOut[i] = swp(hIn[i]) - iv[(i + 1) % 5];
	}
}

/**
Copies the target hashes to constant memory
*/
cudaError_t CudaHashLookup::setTargetConstantMemory(const std::vector<struct hash160> &targets)
{
	size_t count = targets.size();

	for(size_t i = 0; i < count; i++) {
		unsigned int h[5];

		undoRMD160FinalRound(targets[i].h, h);

		cudaError_t err = cudaMemcpyToSymbol(_TARGET_HASH, h, sizeof(unsigned int) * 5, i * sizeof(unsigned int) * 5);

		if(err) {
			return err;
		}
	}

	cudaError_t err = cudaMemcpyToSymbol(_NUM_TARGET_HASHES, &count, sizeof(unsigned int));
	if(err) {
		return err;
	}

	unsigned int useBloomFilter = 0;

	err = cudaMemcpyToSymbol(_USE_BLOOM_FILTER, &useBloomFilter, sizeof(bool));
	if(err) {
		return err;
	}

	return cudaSuccess;
}

/**
Returns the optimal bloom filter size in bits given the probability of false-positives and the
number of hash functions
*/
unsigned int CudaHashLookup::getOptimalBloomFilterBits(double p, size_t n)
{
	double m = 3.6 * ceil((n * log(p)) / log(1 / pow(2, log(2))));

	return (unsigned int)ceil(log(m) / log(2));
}

void CudaHashLookup::initializeBloomFilter(const std::vector<struct hash160> &targets, unsigned int *filter, unsigned int mask)
{
	// Use the low 16 bits of each word in the hash as the index into the bloom filter
	for(unsigned int i = 0; i < targets.size(); i++) {

		unsigned int h[5];

		undoRMD160FinalRound(targets[i].h, h);

		for(int j = 0; j < 5; j++) {
			unsigned int idx = h[j] & mask;

			filter[idx / 32] |= (0x01 << (idx % 32));
		}

	}
}

void CudaHashLookup::initializeBloomFilter64(const std::vector<struct hash160> &targets, unsigned int *filter, unsigned long long mask)
{
	for(unsigned int k = 0; k < targets.size(); k++) {

		unsigned int hash[5];

		unsigned long long idx[5];

		undoRMD160FinalRound(targets[k].h, hash);

		idx[0] = ((unsigned long long)hash[0] << 32 | hash[1]) & mask;
		idx[1] = ((unsigned long long)hash[2] << 32 | hash[3]) & mask;
		idx[2] = ((unsigned long long)(hash[0]^hash[1]) << 32 | (hash[1]^hash[2])) & mask;
		idx[3] = ((unsigned long long)(hash[2]^hash[3]) << 32 | (hash[3] ^ hash[4])) & mask;
		idx[4] = ((unsigned long long)(hash[0]^hash[3]) << 32 | (hash[1]^hash[3])) & mask;

		for(int i = 0; i < 5; i++) {

			filter[idx[i] / 32] |= (0x01 << (idx[i] % 32));
		}
	}
}

/**
Populates the bloom filter with the target hashes
*/
cudaError_t CudaHashLookup::setTargetBloomFilter(const std::vector<struct hash160> &targets)
{
	unsigned int bloomFilterBits = getOptimalBloomFilterBits(1.0e-9, targets.size());

	unsigned long long bloomFilterSizeWords = (unsigned long long)1 << (bloomFilterBits - 5);
	unsigned long long bloomFilterBytes = (unsigned long long)1 << (bloomFilterBits - 3);
	unsigned long long bloomFilterMask = (((unsigned long long)1 << bloomFilterBits) - 1);

	Logger::log(LogLevel::Info, "Allocating bloom filter (" + util::format("%.1f", (double)bloomFilterBytes/(double)(1024*1024)) + "MB)");

	unsigned int *filter = NULL;
	
	try {
		filter = new unsigned int[bloomFilterSizeWords];
	} catch(std::bad_alloc) {
		Logger::log(LogLevel::Error, "Out of system memory");

		return cudaErrorMemoryAllocation;
	}

	cudaError_t err = cudaMalloc(&_bloomFilterPtr, bloomFilterBytes);

	if(err) {
		Logger::log(LogLevel::Error, "Device error: " + std::string(cudaGetErrorString(err)));
		delete[] filter;
		return err;
	}

	memset(filter, 0, sizeof(unsigned int) * bloomFilterSizeWords);
	if(bloomFilterBits > 32) {
		initializeBloomFilter64(targets, filter, bloomFilterMask);
	} else {
		initializeBloomFilter(targets, filter, (unsigned int)bloomFilterMask);
	}

	// Copy to device
	err = cudaMemcpy(_bloomFilterPtr, filter, sizeof(unsigned int) * bloomFilterSizeWords, cudaMemcpyHostToDevice);
	if(err) {
		cudaFree(_bloomFilterPtr);
		_bloomFilterPtr = NULL;
		delete[] filter;
		return err;
	}

	// Copy device memory pointer to constant memory
	err = cudaMemcpyToSymbol(_BLOOM_FILTER, &_bloomFilterPtr, sizeof(unsigned int *));
	if(err) {
		cudaFree(_bloomFilterPtr);
		_bloomFilterPtr = NULL;
		delete[] filter;
		return err;
	}

	// Copy device memory pointer to constant memory
	if(bloomFilterBits <= 32) {
		err = cudaMemcpyToSymbol(_BLOOM_FILTER_MASK, &bloomFilterMask, sizeof(unsigned int));
		if(err) {
			cudaFree(_bloomFilterPtr);
			_bloomFilterPtr = NULL;
			delete[] filter;
			return err;
		}
	} else {
		err = cudaMemcpyToSymbol(_BLOOM_FILTER_MASK64, &bloomFilterMask, sizeof(unsigned long long));
		if(err) {
			cudaFree(_bloomFilterPtr);
			_bloomFilterPtr = NULL;
			delete[] filter;
			return err;
		}
	}

	unsigned int useBloomFilter = bloomFilterBits <= 32 ? 1 : 2;

	err = cudaMemcpyToSymbol(_USE_BLOOM_FILTER, &useBloomFilter, sizeof(unsigned int));

	delete[] filter;

	return err;
}

/**
*Copies the target hashes to either constant memory, or the bloom filter depending
on how many targets there are
*/
cudaError_t CudaHashLookup::setTargets(const std::vector<struct hash160> &targets)
{
	cleanup();

	if(targets.size() <= MAX_TARGETS_CONSTANT_MEM) {
		return setTargetConstantMemory(targets);
	} else {
		return setTargetBloomFilter(targets);
	}
}

void CudaHashLookup::cleanup()
{
	if(_bloomFilterPtr != NULL) {
		cudaFree(_bloomFilterPtr);
		_bloomFilterPtr = NULL;
	}
}

__device__ bool checkBloomFilter(const unsigned int hash[5])
{
	bool foundMatch = true;

	unsigned int mask = _BLOOM_FILTER_MASK[0];
	unsigned int *bloomFilter = _BLOOM_FILTER[0];

	for(int i = 0; i < 5; i++) {
        unsigned int idx = hash[i] & mask;

        unsigned int f = bloomFilter[idx / 32];

		if((f & (0x01 << (idx % 32))) == 0) {
			foundMatch = false;
		}
	}

	return foundMatch;
}

__device__ bool checkBloomFilter64(const unsigned int hash[5])
{
	bool foundMatch = true;

	unsigned long long mask = _BLOOM_FILTER_MASK64[0];
	unsigned int *bloomFilter = _BLOOM_FILTER[0];
	unsigned long long idx[5];

	idx[0] = ((unsigned long long)hash[0] << 32 | hash[1]) & mask;
	idx[1] = ((unsigned long long)hash[2] << 32 | hash[3]) & mask;
	idx[2] = ((unsigned long long)(hash[0] ^ hash[1]) << 32 | (hash[1] ^ hash[2])) & mask;
	idx[3] = ((unsigned long long)(hash[2] ^ hash[3]) << 32 | (hash[3] ^ hash[4])) & mask;
	idx[4] = ((unsigned long long)(hash[0] ^ hash[3]) << 32 | (hash[1] ^ hash[3])) & mask;

	for(int i = 0; i < 5; i++) {
		unsigned int f = bloomFilter[idx[i] / 32];

		if((f & (0x01 << (idx[i] % 32))) == 0) {
			foundMatch = false;
		}
	}

	return foundMatch;
}


__device__ bool checkHash(const unsigned int hash[5])
{
	bool foundMatch = false;

	if(*_USE_BLOOM_FILTER == 1) {
		return checkBloomFilter(hash);
	} else if(*_USE_BLOOM_FILTER == 2) {
		return checkBloomFilter64(hash);
	} else {
		for(int j = 0; j < *_NUM_TARGET_HASHES; j++) {
			bool equal = true;
			for(int i = 0; i < 5; i++) {
				equal &= (hash[i] == _TARGET_HASH[j][i]);
			}

			foundMatch |= equal;
		}
	}

	return foundMatch;
}