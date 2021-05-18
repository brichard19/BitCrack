#ifndef _BRIDGE_H
#define _BRIDGE_H

#include<cuda.h>
#include<cuda_runtime.h>
#include<string>
#include "cudaUtil.h"
#include "secp256k1.h"


void callKeyFinderKernel(int blocks, int threads, int points, unsigned int *xPtr, unsigned int *yPtr, unsigned int *chainPtr, bool useDouble, int compression);

void checkKernelLaunch();
void waitForKernel();

cudaError_t setIncrementorPoint(const secp256k1::uint256 &x, const secp256k1::uint256 &y);
cudaError_t allocateChainBuf(unsigned int count);
void cleanupChainBuf();

#endif
