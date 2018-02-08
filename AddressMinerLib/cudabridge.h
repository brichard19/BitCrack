#ifndef _BRIDGE_H
#define _BRIDGE_H

#include<cuda.h>
#include<cuda_runtime.h>
#include<string>

#include "DeviceContext.h"


class CudaException {

public:

	std::string msg;

	int errCode;

	CudaException(const std::string &msg)
	{
		this->msg = msg;
	}

	CudaException(cudaError_t cudaError)
	{
		errCode = cudaError;

		this->msg = std::string(cudaGetErrorString(cudaError));
	}

};

void callAddressMinerKernel(KernelParams &params);

void waitForKernel();

cudaError_t setMinMaxTarget(const unsigned int *min, const unsigned int *max, const unsigned int *qx, const unsigned int *qy);


#endif