#ifndef _EC_CUH
#define _EC_CUH

#include <cuda_runtime.h>

namespace ec {
	__device__ unsigned int *getXPtr();

	__device__ unsigned int *getYPtr();
}

#endif