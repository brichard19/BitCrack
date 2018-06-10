#include<stdio.h>

#include "DeviceContext.h"


static std::string getErrorString(cudaError_t err)
{
	return std::string(cudaGetErrorString(err));
}

void DeviceContext::init(int device, int threads, int blocks, int pointsPerThread)
{
	_device = device;
	_threads = threads;
	_blocks = blocks;
	_pointsPerThread = pointsPerThread;

	this->_device = device;

	unsigned int count = _pointsPerThread * _threads * _blocks;

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

	// Allocate chain buf
	err = cudaMalloc(&_chain, sizeof(unsigned int) * count * 8);
	if(err) {
		goto end;
	}

	_numResultsHost = NULL;
	err = cudaHostAlloc(&_numResultsHost, sizeof(unsigned int), cudaHostAllocMapped);
	if(err) {
		goto end;
	}

	_numResultsDev = NULL;
	err = cudaHostGetDevicePointer(&_numResultsDev, _numResultsHost, 0);
	if(err) {
		goto end;
	}
	*_numResultsHost = 0;


	_resultsHost = NULL;
	err = cudaHostAlloc(&_resultsHost, 4096, cudaHostAllocMapped);
	if(err) {
		goto end;
	}

	_resultsDev = NULL;
	err = cudaHostGetDevicePointer(&_resultsDev, _resultsHost, 0);
	if(err) {
		goto end;
	}

end:
	if(err) {
		cudaFree(_x);
		cudaFree(_y);
		cudaFree(_chain);
		cudaFreeHost(_numResultsHost);
		cudaFree(_numResultsDev);
		cudaFreeHost(_resultsHost);
		cudaFree(_resultsDev);
		throw DeviceContextException(getErrorString(err));
	}
}

void DeviceContext::copyPoints(const std::vector<secp256k1::ecpoint> &points)
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

int DeviceContext::getIndex(int block, int thread, int idx)
{
	// Total number of threads
	int totalThreads = _blocks * _threads;

	int base = idx * totalThreads;

	// Global ID of the current thread
	int threadId = block * _threads + thread;

	return base + threadId;
}

void DeviceContext::splatBigInt(unsigned int *dest, int block, int thread, int idx, const secp256k1::uint256 &i)
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

void DeviceContext::cleanup()
{
	if(_x != NULL) {
		cudaFree(_y);
	}

	if(_y != NULL) {
		cudaFree(_y);
	}

	if(_chain != NULL) {
		cudaFree(_chain);
	}

	if(_numResultsHost != NULL) {
		cudaFreeHost(_numResultsHost);
	}

	if(_resultsHost != NULL) {
		cudaFreeHost(_resultsHost);
	}

	_x = NULL;
	_y = NULL;
	_chain = NULL;

	cudaDeviceReset();
}


KernelParams DeviceContext::getKernelParams()
{
	KernelParams params;
	params.blocks = _blocks;
	params.threads = _threads;
	params.points = _pointsPerThread;
	params.x = _x;
	params.y = _y;
	params.chain = _chain;

	params.results = _resultsDev;
	params.numResults = _numResultsDev;
	params.flags = PointCompressionType::UNCOMPRESSED | PointCompressionType::COMPRESSED;

	return params;
}

void DeviceContext::getKeyFinderResults(std::vector<struct KeyFinderResult> &results)
{
	results.clear();

	int numResults = *_numResultsHost;

	for(int i = 0; i < numResults; i++) {
		struct KeyFinderDeviceResult *ptr = &((struct KeyFinderDeviceResult *)_resultsHost)[i];

		KeyFinderResult minerResult;
		minerResult.block = ptr->block;
		minerResult.thread = ptr->thread;
		minerResult.index = ptr->idx;
		minerResult.compressed = ptr->compressed;
		for(int i = 0; i < 5; i++) {
			minerResult.hash[i] = ptr->digest[i];
		}
		minerResult.p = secp256k1::ecpoint(secp256k1::uint256(ptr->x, secp256k1::uint256::BigEndian), secp256k1::uint256(ptr->y, secp256k1::uint256::BigEndian));

		results.push_back(minerResult);
	}

	// Clear results
	*_numResultsHost = 0;
}

bool DeviceContext::resultFound()
{
	unsigned int numResults = (*_numResultsHost) != 0;

	return numResults != 0;
}

DeviceContext::~DeviceContext()
{
	cleanup();
}