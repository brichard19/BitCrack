#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "KeySearchTypes.h"
#include "CKSD.h"
#include "ptx.cuh"
#include "secp256k1.cuh"

#include "sha256.cuh"
#include "ripemd160.cuh"

#include "secp256k1.h"

#include "CudaHashLookup.cuh"
#include "CudaAtomicList.cuh"
#include "CudaDeviceKeys.cuh"

__constant__ unsigned int _INC_X[8];

__constant__ unsigned int _INC_Y[8];
