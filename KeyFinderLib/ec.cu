#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "ec.h"
#include "ec.cuh"
#include "secp256k1.cuh"


__constant__ unsigned int *_xPtr[1];

__constant__ unsigned int *_yPtr[1];


__device__ unsigned int *ec::getXPtr()
{
	return _xPtr[0];
}

__device__ unsigned int *ec::getYPtr()
{
	return _yPtr[0];
}

__global__ void multiplyStepKernel(const unsigned int *privateKeys, int pointsPerThread, int step, unsigned int *chain, const unsigned int *gxPtr, const unsigned int *gyPtr);


int CudaDeviceKeys::getIndex(int block, int thread, int idx)
{
	// Total number of threads
	int totalThreads = _blocks * _threads;

	int base = idx * totalThreads;

	// Global ID of the current thread
	int threadId = block * _threads + thread;

	return base + threadId;
}

void CudaDeviceKeys::splatBigInt(unsigned int *dest, int block, int thread, int idx, const secp256k1::uint256 &i)
{
	unsigned int value[8] = { 0 };

	i.exportWords(value, 8, secp256k1::uint256::BigEndian);

	int totalThreads = _blocks * _threads;
	int threadId = block * _threads + thread;

	int base = idx * _blocks * _threads * 8;

	int index = base + threadId;

	for(int k = 0; k < 8; k++) {
		dest[index] = value[k];
		index += totalThreads;
	}
}

secp256k1::uint256 CudaDeviceKeys::readBigInt(unsigned int *src, int block, int thread, int idx)
{
	unsigned int value[8] = { 0 };

	int totalThreads = _blocks * _threads;
	int threadId = block * _threads + thread;

	int base = idx * _blocks * _threads * 8;

	int index = base + threadId;

	for(int k = 0; k < 8; k++) {
		value[k] = src[index];
		index += totalThreads;
	}

	secp256k1::uint256 v(value, secp256k1::uint256::BigEndian);

	return v;
}

/**
* Allocates device memory for storing the multiplication chain used in
the batch inversion operation
*/
cudaError_t CudaDeviceKeys::allocateChainBuf(unsigned int count)
{
	cudaError_t err = cudaMalloc(&_devChain, count * sizeof(unsigned int) * 8);

	if(err) {
		return err;
	}

	return err;
}

cudaError_t CudaDeviceKeys::initializeBasePoints()
{
	// generate a table of points G, 2G, 4G, 8G...(2^255)G
	std::vector<secp256k1::ecpoint> table;

	table.push_back(secp256k1::G());
	for(int i = 1; i < 256; i++) {

		secp256k1::ecpoint p = doublePoint(table[i - 1]);
		if(!pointExists(p)) {
			throw std::string("Point does not exist!");
		}
		table.push_back(p);
	}

	unsigned int count = 256;

	cudaError_t err = cudaMalloc(&_devBasePointX, sizeof(unsigned int) * count * 8);

	if(err) {
		return err;
	}

	err = cudaMalloc(&_devBasePointY, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	unsigned int *tmpX = new unsigned int[count * 8];
	unsigned int *tmpY = new unsigned int[count * 8];

	for(int i = 0; i < 256; i++) {
		unsigned int bufX[8];
		unsigned int bufY[8];
		table[i].x.exportWords(bufX, 8, secp256k1::uint256::BigEndian);
		table[i].y.exportWords(bufY, 8, secp256k1::uint256::BigEndian);

		for(int j = 0; j < 8; j++) {
			tmpX[i * 8 + j] = bufX[j];
			tmpY[i * 8 + j] = bufY[j];
		}
	}

	err = cudaMemcpy(_devBasePointX, tmpX, count * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);

	delete[] tmpX;

	if(err) {
		delete[] tmpY;
		return err;
	}

	err = cudaMemcpy(_devBasePointY, tmpY, count * 8 * sizeof(unsigned int), cudaMemcpyHostToDevice);

	delete[] tmpY;

	return err;
}

cudaError_t CudaDeviceKeys::initializePublicKeys(unsigned int count)
{

	// Allocate X array
	cudaError_t err = cudaMalloc(&_devX, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	// Clear X array
	err = cudaMemset(_devX, -1, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	// Allocate Y array
	err = cudaMalloc(&_devY, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	// Clear Y array
	err = cudaMemset(_devY, -1, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	err = cudaMemcpyToSymbol(_xPtr, &_devX, sizeof(unsigned int *));
	if(err) {
		return err;
	}

	err = cudaMemcpyToSymbol(_yPtr, &_devY, sizeof(unsigned int *));
	
	return err;
}

cudaError_t CudaDeviceKeys::init(int blocks, int threads, int pointsPerThread, const std::vector<secp256k1::uint256> &privateKeys)
{
	_blocks = blocks;
	_threads = threads;
	_pointsPerThread = pointsPerThread;

	unsigned int count = privateKeys.size();

	// Allocate space for public keys on device
	cudaError_t err = initializePublicKeys(count);

	if(err) {
		return err;
	}

	err = initializeBasePoints();
	if(err) {
		return err;
	}

	// Allocate private keys on device
	err = cudaMalloc(&_devPrivate, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}


	// Clear private keys
	err = cudaMemset(_devPrivate, 0, sizeof(unsigned int) * count * 8);
	if(err) {
		return err;
	}

	err = allocateChainBuf(_threads * _blocks * _pointsPerThread);
	if(err) {
		return err;
	}

	// Copy private keys to system memory buffer
	unsigned int *tmp = new unsigned int[count * 8];

	for(int block = 0; block < _blocks; block++) {
		for(int thread = 0; thread < _threads; thread++) {
			for(int idx = 0; idx < _pointsPerThread; idx++) {

				int index = getIndex(block, thread, idx);

				splatBigInt(tmp, block, thread, idx, privateKeys[index]);
			}
		}
	}

	// Copy private keys to device memory
	err = cudaMemcpy(_devPrivate, tmp, count * sizeof(unsigned int) * 8, cudaMemcpyHostToDevice);

	delete[] tmp;

	if(err) {
		return err;
	}

	return cudaSuccess;
}

void CudaDeviceKeys::clearPublicKeys()
{
	cudaFree(_devX);
	cudaFree(_devY);

	_devX = NULL;
	_devY = NULL;
}

void CudaDeviceKeys::clearPrivateKeys()
{
	cudaFree(_devBasePointX);
	cudaFree(_devBasePointY);
	cudaFree(_devPrivate);
	cudaFree(_devChain);

	_devChain = NULL;
	_devBasePointX = NULL;
	_devBasePointY = NULL;
	_devPrivate = NULL;
}

cudaError_t CudaDeviceKeys::doStep()
{
	multiplyStepKernel <<<_blocks, _threads>>>(_devPrivate, _pointsPerThread, _step, _devChain, _devBasePointX, _devBasePointY);

	// Wait for kernel to complete
    cudaError_t err = cudaDeviceSynchronize();
	fflush(stdout);
	_step++;
	return err;
}

__global__ void multiplyStepKernel(const unsigned int *privateKeys, int pointsPerThread, int step, unsigned int *chain, const unsigned int *gxPtr, const unsigned int *gyPtr)
{
	unsigned int *xPtr = ec::getXPtr();

	unsigned int *yPtr = ec::getYPtr();

	unsigned int gx[8];
	unsigned int gy[8];

	for(int i = 0; i < 8; i++) {
		gx[i] = gxPtr[step * 8 + i];
		gy[i] = gyPtr[step * 8 + i];
	}

	// Multiply together all (_Gx - x) and then invert
	unsigned int inverse[8] = { 0,0,0,0,0,0,0,1 };

	int batchIdx = 0;
	for(int i = 0; i < pointsPerThread; i++) {

		unsigned int p[8];
		readInt(privateKeys, i, p);
		unsigned int bit = p[7 - step / 32] & 1 << ((step % 32));
		
		unsigned int x[8];
		readInt(xPtr, i, x);

		if(bit != 0) {
			if(!isInfinity(x)) {
				beginBatchAddWithDouble(gx, gy, xPtr, chain, i, batchIdx, inverse);
				batchIdx++;
			}
		}
	}

	doBatchInverse(inverse);

	for(int i = pointsPerThread - 1; i >= 0; i--) {

		unsigned int newX[8];
		unsigned int newY[8];

		unsigned int p[8];
		readInt(privateKeys, i, p);
		unsigned int bit = p[7 - step / 32] & 1 << ((step % 32));

		unsigned int x[8];
		readInt(xPtr, i, x);

		bool infinity = isInfinity(x);

		if(bit != 0) {
			if(!infinity) {
				batchIdx--;
				completeBatchAddWithDouble(gx, gy, xPtr, yPtr, i, batchIdx, chain, inverse, newX, newY);
			} else {
				copyBigInt(gx, newX);
				copyBigInt(gy, newY);
			}

			writeInt(xPtr, i, newX);
			writeInt(yPtr, i, newY);
		}
	}
}

bool CudaDeviceKeys::selfTest(const std::vector<secp256k1::uint256> &privateKeys)
{
	unsigned int numPoints = _threads * _blocks * _pointsPerThread;

	unsigned int *xBuf = new unsigned int[numPoints * 8];
	unsigned int *yBuf = new unsigned int[numPoints * 8];

	cudaError_t err = cudaMemcpy(xBuf, _devX, sizeof(unsigned int) * 8 * numPoints, cudaMemcpyDeviceToHost);

	err = cudaMemcpy(yBuf, _devY, sizeof(unsigned int) * 8 * numPoints, cudaMemcpyDeviceToHost);


	for(int block = 0; block < _blocks; block++) {
		for(int thread = 0; thread < _threads; thread++) {
			for(int idx = 0; idx < _pointsPerThread; idx++) {

				int index = getIndex(block, thread, idx);

				secp256k1::uint256 privateKey = privateKeys[index];

				secp256k1::uint256 x = readBigInt(xBuf, block, thread, idx);
				secp256k1::uint256 y = readBigInt(yBuf, block, thread, idx);

				secp256k1::ecpoint p1(x, y);
				secp256k1::ecpoint p2 = secp256k1::multiplyPoint(privateKey, secp256k1::G());

				if(!secp256k1::pointExists(p1)) {
					throw std::string("Validation failed: invalid point");
				}

				if(!secp256k1::pointExists(p2)) {
					throw std::string("Validation failed: invalid point");
				}

				if(!(p1 == p2)) {
					throw std::string("Validation failed: points do not match");
				}
			}
		}
	}

	return true;
}