#include<stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cudaUtil.h"
#include "CudaDeviceContext.h"


static std::string getErrorString(cudaError_t err)
{
	return std::string(cudaGetErrorString(err));
}

CudaDeviceContext::CudaDeviceContext(const DeviceParameters &params)
{
	_device = params.device;
	_threads = params.threads;
	_blocks = params.blocks;
	_pointsPerThread = params.pointsPerThread;

	try {
		cuda::CudaDeviceInfo device = cuda::getDeviceInfo(_device);

		_deviceName = device.name;
	} catch(cuda::CudaException ex) {
		throw DeviceContextException(ex.msg);
	}

	_x = NULL;
	_y = NULL;
}

void CudaDeviceContext::init()
{
	size_t count = _pointsPerThread * _threads * _blocks;

	cudaError_t err = cudaSetDevice(_device);

	if(err != cudaSuccess) {
		goto end;
	}

	// Allocate X array
	err = cudaMalloc(&_x, sizeof(unsigned int) * count * 8);
	if(err != cudaSuccess) {
		goto end;
	}

	// Clear X array
	err = cudaMemset(_x, 0, sizeof(unsigned int) * count * 8);
	if(err != cudaSuccess) {
		goto end;
	}

	// Allocate Y array
	err = cudaMalloc(&_y, sizeof(unsigned int) * count * 8);
	if(err != cudaSuccess) {
		goto end;
	}

	// Clear Y array
	err = cudaMemset(_y, 0, sizeof(unsigned int) * count * 8);
	if(err != cudaSuccess) {
		goto end;
	}

end:
	if(err) {
		cudaFree(_x);
		cudaFree(_y);

		throw DeviceContextException(getErrorString(err));
	}
}

void CudaDeviceContext::copyPoints(const std::vector<secp256k1::ecpoint> &points)
{
	int count = _pointsPerThread * _threads * _blocks * 8;
	unsigned int *tmpX = new unsigned int[count];
	unsigned int *tmpY = new unsigned int[count];

	unsigned int totalThreads = _blocks * _threads;

	for(int block = 0; block < _blocks; block++) {
		for(int thread = 0; thread < _threads; thread++) {
			for(int idx = 0; idx < _pointsPerThread; idx++) {

				int index = getIndex(block, thread, idx);

				splatBigInt(tmpX, block, thread, idx, points[index].x);
				splatBigInt(tmpY, block, thread, idx, points[index].y);

			}
		}
	}

	cudaError_t err = cudaMemcpy(_x, tmpX, sizeof(unsigned int) * count, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		delete[] tmpX;
		delete[] tmpY;
		throw DeviceContextException(getErrorString(err));
	}

	err = cudaMemcpy(_y, tmpY, sizeof(unsigned int) * count, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
		delete[] tmpX;
		delete[] tmpY;
		throw DeviceContextException(getErrorString(err));
	}

	delete[] tmpX;
	delete[] tmpY;
}

int CudaDeviceContext::getIndex(int block, int thread, int idx)
{
	// Total number of threads
	int totalThreads = _blocks * _threads;

	int base = idx * totalThreads;

	// Global ID of the current thread
	int threadId = block * _threads + thread;

	return base + threadId;
}

void CudaDeviceContext::splatBigInt(unsigned int *dest, int block, int thread, int idx, const secp256k1::uint256 &i)
{
	unsigned int value[8] = { 0 };

	i.exportWords(value, 8);

	int totalThreads = _blocks * _threads;
	int threadId = block * _threads + thread;

	int base = idx * _blocks * _threads * 8;

	int index = base + threadId;

	for(int k = 0; k < 8; k++) {
		dest[index] = value[7 - k];
		index += totalThreads;
	}
}

void CudaDeviceContext::cleanup()
{
	cudaFree(_y);

	cudaFree(_y);

	_x = NULL;
	_y = NULL;

	cudaDeviceReset();
}


KernelParams CudaDeviceContext::getKernelParams()
{
	KernelParams params;
	params.blocks = _blocks;
	params.threads = _threads;
	params.points = _pointsPerThread;
	params.x = _x;
	params.y = _y;

	return params;
}

CudaDeviceContext::~CudaDeviceContext()
{
	cleanup();
}

void CudaDeviceContext::getMemInfo(size_t &freeMem, size_t &totalMem)
{
	cudaError_t err = cudaMemGetInfo(&freeMem, &totalMem);

	if(err) {
		throw DeviceContextException(getErrorString(err));
	}
}

std::string CudaDeviceContext::getDeviceName()
{
	return _deviceName;
}