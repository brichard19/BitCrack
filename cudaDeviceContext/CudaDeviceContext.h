#ifndef _CUDA_DEVICE_CONTEXT_H
#define _CUDA_DEVICE_CONTEXT_H

#include "DeviceContext.h"

class CudaDeviceContext : DeviceContext {

private:
	int _device;

	int _threads;
	int _blocks;
	int _pointsPerThread;

	unsigned int *_x;
	unsigned int *_y;

	unsigned int *_numResultsHost;
	unsigned int *_numResultsDev;
	unsigned int *_resultsHost;
	unsigned int *_resultsDev;

	void splatBigInt(unsigned int *dest, int block, int thread, int idx, const secp256k1::uint256 &value);

public:
	CudaDeviceContext()
	{
		_device = 0;
		_threads = 0;
		_blocks = 0;
		_pointsPerThread = 0;

		_x = NULL;
		_y = NULL;
	}

	void init(const DeviceParameters &params);

	void copyPoints(const std::vector<secp256k1::ecpoint> &points);
	int getIndex(int block, int thread, int idx);

	KernelParams getKernelParams();

	void cleanup();

	bool resultFound();
	int getResultCount();
	void getResults(void *ptr, int size);
	void clearResults();
	~CudaDeviceContext();
};

#endif