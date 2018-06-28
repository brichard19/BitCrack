#ifndef _DEVICE_CONTEXT_H
#define _DEVICE_CONTEXT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "secp256k1.h"
#include "DeviceContextShared.h"

struct KeyFinderResult {
	int thread;
	int block;
	int index;
	bool compressed;

	secp256k1::ecpoint p;
	unsigned int hash[5];
};

typedef struct {
	int blocks;
	int threads;
	int points;

	unsigned int *x;
	unsigned int *y;

	unsigned int *results;
	unsigned int *numResults;

}KernelParams;

typedef struct {
	int device;
	int blocks;
	int threads;
	int pointsPerThread;
}DeviceParameters;

class DeviceContextException {

public:

	DeviceContextException(const std::string &msg)
	{
		this->msg = msg;
	}

	std::string msg;
};

class DeviceContext {

public:

	virtual void init(const DeviceParameters &params) = 0;
	virtual void copyPoints(const std::vector<secp256k1::ecpoint> &points) = 0;

	virtual KernelParams getKernelParams() = 0;

	virtual void cleanup() = 0;

	virtual bool resultFound() = 0;
	virtual int getResultCount() = 0;
	virtual void getResults(void *ptr, int size) = 0;
	virtual void clearResults() = 0;

	virtual ~DeviceContext() {}
};

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