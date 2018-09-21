#ifndef _BRIDGE_H
#define _BRIDGE_H

#include<cuda.h>
#include<cuda_runtime.h>
#include<string>

#include "DeviceContext.h"
#include "cudaUtil.h"


void callKeyFinderKernel(KernelParams &params, bool useDouble, int compression);

void waitForKernel();

cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y);
cudaError_t allocateChainBuf(unsigned int count);
void cleanupChainBuf();

#endif