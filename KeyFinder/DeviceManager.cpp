#include "DeviceManager.h"

#ifdef BUILD_CUDA
#include "cudaUtil.h"
#endif

#ifdef BUILD_OPENCL
#include "clUtil.h"
#endif

std::vector<DeviceManager::DeviceInfo> DeviceManager::getDevices()
{
    int deviceId = 0;

    std::vector<DeviceManager::DeviceInfo> devices;

#ifdef BUILD_CUDA
    // Get CUDA devices
    try {
        std::vector<cuda::CudaDeviceInfo> cudaDevices = cuda::getDevices();

        for(int i = 0; i < cudaDevices.size(); i++) {
            DeviceManager::DeviceInfo device;
            device.name = cudaDevices[i].name;
            device.type = DeviceType::CUDA;
            device.id = deviceId;
            device.physicalId = cudaDevices[i].id;

            devices.push_back(device);

            deviceId++;
        }
    } catch(cuda::CudaException ex) {
        throw DeviceManager::DeviceManagerException(ex.msg);
    }
#endif

#ifdef BUILD_OPENCL
    // Get OpenCL devices
    try {
        std::vector<cl::CLDeviceInfo> clDevices = cl::getDevices();

        for(int i = 0; i < clDevices.size(); i++) {
            DeviceManager::DeviceInfo device;
            device.name = clDevices[i].name;
            device.type = DeviceType::OpenCL;
            device.id = deviceId;
            device.physicalId = clDevices[i].id;

            devices.push_back(device);
            deviceId++;
        }
    } catch(cl::CLException ex) {
        throw DeviceManager::DeviceManagerException(ex.msg);
    }
#endif

    return devices;
}