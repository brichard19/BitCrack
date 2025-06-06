#ifndef _HASH_LOOKUP_HOST_H
#define _HASH_LOOKUP_HOST_H

#include <cuda_runtime.h>
#include <vector>   // Required for std::vector
#include <cstddef>  // Required for size_t

// hash160 struct is defined in KeySearchTypes.h
#include "../KeyFinderLib/KeySearchTypes.h"

class CudaHashLookup {

private:
	unsigned int *_bloomFilterPtr;

	cudaError_t setTargetBloomFilter(const std::vector<struct hash160> &targets);
	
	cudaError_t setTargetConstantMemory(const std::vector<struct hash160> &targets);
	
	unsigned int getOptimalBloomFilterBits(double p, size_t n);

	void cleanup();

	void initializeBloomFilter(const std::vector<struct hash160> &targets, unsigned int *filter, unsigned int mask);
	
	void initializeBloomFilter64(const std::vector<struct hash160> &targets, unsigned int *filter, unsigned long long mask);

public:

	CudaHashLookup()
	{
		_bloomFilterPtr = NULL;
	}

	~CudaHashLookup()
	{
		cleanup();
	}

	cudaError_t setTargets(const std::vector<struct hash160> &targets);
};

#endif