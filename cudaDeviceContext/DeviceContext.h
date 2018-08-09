#ifndef _DEVICE_CONTEXT_H
#define _DEVICE_CONTEXT_H


#include <vector>
#include "secp256k1.h"

typedef struct {
	int blocks;
	int threads;
	int points;

	unsigned int *x;
	unsigned int *y;

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

	virtual void init() = 0;

	virtual void copyPoints(const std::vector<secp256k1::ecpoint> &points) = 0;

	virtual KernelParams getKernelParams() = 0;

	virtual void cleanup() = 0;

	virtual void getMemInfo(size_t &freeMem, size_t &totalMem) = 0;

	virtual std::string getDeviceName() = 0;

	virtual ~DeviceContext() {}
};

#endif