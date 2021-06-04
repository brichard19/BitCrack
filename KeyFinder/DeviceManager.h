#ifndef DEVICE_MANAGER_H
#define DEVICE_MANAGER_H

#include <stdint.h>
#include <string>
#include <vector>

namespace DeviceManager {

class DeviceManagerException {

public:
    std::string msg;

    DeviceManagerException(const std::string &msg)
    {
        this->msg = msg;
    }
};

typedef struct {
    int id;

    // General device info
    uint64_t physicalId;
    std::string name;
    uint64_t memory;
    int computeUnits;

}DeviceInfo;

std::vector<DeviceInfo> getDevices();

}

#endif
