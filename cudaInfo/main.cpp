#include <stdio.h>
#include <vector>

#include"cudaUtil.h"

void printDeviceInfo(const cuda::CudaDeviceInfo &info)
{
	printf("ID:          %d\n", info.id);
	printf("Name:        %s\n", info.name.c_str());
	printf("Capability:  %d.%d\n", info.major, info.minor);
	printf("MP:          %d\n", info.mpCount);
	printf("Cores:       %d (%d per MP)\n", info.mpCount * info.cores, info.cores);
	printf("Memory:      %dMB\n", (int)(info.mem / (1024 * 1024)));
}

int main(int argc, char **argv)
{
	std::vector<cuda::CudaDeviceInfo> devices = cuda::getDevices();

	printf("Found %d devices\n\n", (int)devices.size());

	for(int i = 0; i < (int)devices.size(); i++) {
		printDeviceInfo(devices[i]);
		printf("\n");
	}

	return 0;
}