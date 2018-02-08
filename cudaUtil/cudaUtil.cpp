#include "cudaUtil.h"


cuda::CudaDeviceInfo cuda::getDeviceInfo(int device)
{
	cuda::CudaDeviceInfo devInfo;

	cudaDeviceProp properties;
	cudaError_t err = cudaSuccess;

	err = cudaSetDevice(device);

	if(err) {
		throw cuda::CudaException(err);
	}

	err = cudaGetDeviceProperties(&properties, device);
	
	if(err) {
		throw cuda::CudaException(err);
	}

	devInfo.major = properties.major;
	devInfo.minor = properties.minor;
	devInfo.mpCount = properties.multiProcessorCount;
	devInfo.mem = properties.totalGlobalMem;
	devInfo.name = std::string(properties.name);

	int cores = 0;
	switch(devInfo.major) {
	case 1:
		cores = 8;
		break;
	case 2:
		cores = devInfo.minor == 0 ? 32 : 48;
		break;
	case 3:
		cores = 192;
		break;
	case 5:
		cores = 128;
		break;
	case 6:
	case 7:
		cores = 64;
		break;
	}
	devInfo.cores = cores;

	return devInfo;
}


std::vector<cuda::CudaDeviceInfo> cuda::getDevices()
{
	int count = getDeviceCount();

	std::vector<CudaDeviceInfo> devList;

	for(int device = 0; device < count; device++) {
		devList.push_back(getDeviceInfo(device));
	}

	return devList;
}

int cuda::getDeviceCount()
{
	int count = 0;
	cudaError_t err = cudaGetDeviceCount(&count);
	if(err) {
		throw cuda::CudaException(err);
	}

	return count;
}