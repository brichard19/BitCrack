#ifndef _DEVICE_CONTEXT_H
#define _DEVICE_CONTEXT_H

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include "secp256k1.h"


struct AddressMinerResult {

	int thread;
	int block;
	int index;
	int autoType;
	bool compressed;

	secp256k1::ecpoint p;
};

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
	unsigned int *chain;

	unsigned int *results;
	unsigned int *numResults;
	unsigned int flags;
}KernelParams;

class DeviceContextException {

public:

	DeviceContextException(const std::string &msg)
	{
		this->msg = msg;
	}

	std::string msg;
};

class DeviceContext {

private:
	int _device;

	int _threads;
	int _blocks;
	int _pointsPerThread;

	unsigned int *_x;
	unsigned int *_y;
	unsigned int *_chain;

	unsigned int *_numResultsHost;
	unsigned int *_numResultsDev;
	unsigned int *_resultsHost;
	unsigned int *_resultsDev;

	void splatBigInt(unsigned int *dest, int block, int thread, int idx, const secp256k1::uint256 &value);

public:
	DeviceContext()
	{
		_device = 0;
		_threads = 0;
		_blocks = 0;
		_pointsPerThread = 0;

		_x = NULL;
		_y = NULL;
		_chain = NULL;
	}

	void init(int device, int threads, int blocks, int pointsPerThread);
	void copyPoints(const std::vector<secp256k1::ecpoint> &points);
	int getIndex(int block, int thread, int idx);

	KernelParams getKernelParams();

	void cleanup();

	bool resultFound();

	void getAddressMinerResults(std::vector<struct AddressMinerResult> &results);

	void getKeyFinderResults(std::vector<struct KeyFinderResult> &results);

	~DeviceContext();
};

#endif