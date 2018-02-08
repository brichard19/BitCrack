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

void callKeyFinderKernel(KernelParams &params, bool useDouble);

void waitForKernel();

cudaError_t setTargetHash(const unsigned int hash[5]);

cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y);

#endif