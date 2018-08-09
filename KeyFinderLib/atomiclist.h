#ifndef _ATOMIC_LIST_HOST_H
#define _ATOMIC_LIST_HOST_H

#include <cuda_runtime.h>

class CudaAtomicList {

private:
	void *_devPtr;

	void *_hostPtr;

	unsigned int *_countHostPtr;

	unsigned int *_countDevPtr;

	unsigned int _maxSize;

	unsigned int _itemSize;

public:

	CudaAtomicList()
	{
		_devPtr = NULL;
		_hostPtr = NULL;
		_countHostPtr = NULL;
		_countDevPtr = NULL;
		_maxSize = 0;
		_itemSize = 0;
	}

	~CudaAtomicList()
	{
		cleanup();
	}

	cudaError_t init(unsigned int itemSize, unsigned int maxItems);

	unsigned int read(void *dest, unsigned int count);

	unsigned int size();

	void clear();

	void cleanup();

};

#endif