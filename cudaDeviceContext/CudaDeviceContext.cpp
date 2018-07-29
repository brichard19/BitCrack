#include<stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include "CudaDeviceContext.h"

#define RESULTS_BUFFER_SIZE 4 * 1024 * 1024

static std::string getErrorString(cudaError_t err)
{
	return std::string(cudaGetErrorString(err));
}

void CudaDeviceContext::init(const DeviceParameters &params)
{
	_device = params.device;
	_threads = params.threads;
	_blocks = params.blocks;
	_pointsPerThread = params.pointsPerThread;

	
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

	// The number of results found in the most recent kernel run
	_numResultsHost = NULL;
	err = cudaHostAlloc(&_numResultsHost, sizeof(unsigned int), cudaHostAllocMapped);
	if(err) {
		goto end;
	}

	// Number of results found in the most recent kernel run (device to host pointer)
	_numResultsDev = NULL;
	err = cudaHostGetDevicePointer(&_numResultsDev, _numResultsHost, 0);
	if(err) {
		goto end;
	}
	*_numResultsHost = 0;

	// Storage for results data
	_resultsHost = NULL;
	err = cudaHostAlloc(&_resultsHost, RESULTS_BUFFER_SIZE, cudaHostAllocMapped);
	if(err) {
		goto end;
	}

	// Storage for results data (device to host pointer)
	_resultsDev = NULL;
	err = cudaHostGetDevicePointer(&_resultsDev, _resultsHost, 0);
	if(err) {
		goto end;
	}

end:
	if(err) {
		cudaFree(_x);
		cudaFree(_y);
		cudaFreeHost(_numResultsHost);
		cudaFree(_numResultsDev);
		cudaFreeHost(_resultsHost);
		cudaFree(_resultsDev);
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
		throw DeviceContextException(getErrorString(err));
	}

	err = cudaMemcpy(_y, tmpY, sizeof(unsigned int) * count, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) {
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

	cudaFreeHost(_numResultsHost);

	cudaFreeHost(_resultsHost);

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

	params.results = _resultsDev;
	params.numResults = _numResultsDev;

	return params;
}

void CudaDeviceContext::getResults(void *ptr, int size)
{
	memcpy(ptr, _resultsHost, size);
}

int CudaDeviceContext::getResultCount()
{
	return *_numResultsHost;
}

void CudaDeviceContext::clearResults()
{
	*_numResultsHost = 0;
}

bool CudaDeviceContext::resultFound()
{
	unsigned int numResults = (*_numResultsHost) != 0;

	return numResults != 0;
}

CudaDeviceContext::~CudaDeviceContext()
{
	cleanup();
}