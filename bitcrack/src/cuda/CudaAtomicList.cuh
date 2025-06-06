#ifndef _ATOMIC_LIST_CUH
#define _ATOMIC_LIST_CUH

#include <cuda_runtime.h>

__device__ void atomicListAdd(void *info, unsigned int size);

#endif