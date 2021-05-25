#include "DeviceManager.h"
#include "clutil.h"

std::vector<DeviceManager::DeviceInfo> DeviceManager::getDevices()
{
    int deviceId = 0;

    std::vector<DeviceManager::DeviceInfo> devices;

    // Get OpenCL devices
    try {
        std::vector<cl::CLDeviceInfo> clDevices = cl::getDevices();

        for(int i = 0; i < clDevices.size(); i++) {
            DeviceManager::DeviceInfo device;
            device.name = clDevices[i].name;
            device.type = DeviceType::OpenCL;
            device.id = deviceId;
            device.physicalId = (uint64_t)clDevices[i].id;
            device.memory = clDevices[i].mem;
            device.computeUnits = clDevices[i].cores;
            devices.push_back(device);
            deviceId++;
        }
    } catch(cl::CLException ex) {
        throw DeviceManager::DeviceManagerException(ex.msg);
    }

    return devices;
}