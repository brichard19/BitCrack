#ifndef _HASH_LOOKUP_HOST_H
#define _HASH_LOOKUP_HOST_H

#include <cuda_runtime.h>

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