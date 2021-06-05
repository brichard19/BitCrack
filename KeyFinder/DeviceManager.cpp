#include "DeviceManager.h"
#include "clutil.h"

std::vector<DeviceManager::DeviceInfo> DeviceManager::getDevices()
{

    std::vector<DeviceManager::DeviceInfo> devices;

    try {
        std::vector<cl::CLDeviceInfo> clDevices = cl::getDevices();

        for(size_t i = 0; i < clDevices.size(); i++) {
            DeviceManager::DeviceInfo device;
            device.name = clDevices[i].name;
            device.id = i;
            device.physicalId = (uint64_t)clDevices[i].id;
            device.memory = clDevices[i].mem;
            device.computeUnits = clDevices[i].cores;
            device.maxWorkingGroupSize = clDevices[i].maxWorkingGroupSize;
            devices.push_back(device);
        }
    } catch(cl::CLException ex) {
        throw DeviceManager::DeviceManagerException(ex.msg);
    }

    return devices;
}
