#include "clutil.h"

void cl::clCall(cl_int err)
{
    if(err != CL_SUCCESS) {
        throw cl::CLException(err);
    }
}

std::vector<cl::CLDeviceInfo> cl::getDevices()
{
    std::vector<cl::CLDeviceInfo> deviceList;

    cl_uint platformCount = 0;

    clCall(clGetPlatformIDs(0, NULL, &platformCount));

    if(platformCount == 0) {
        return deviceList;
    }

    cl_platform_id* platforms = new cl_platform_id[platformCount];

    clCall(clGetPlatformIDs(platformCount, platforms, NULL));

    for(cl_uint i = 0; i < platformCount; i++) {
        
        cl_uint deviceCount = 0;
        clCall(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, NULL, &deviceCount));

        if(deviceCount == 0) {
            continue;
        }

        cl_device_id* devices = new cl_device_id[deviceCount];
        clCall(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL));

        for(cl_uint j = 0; j < deviceCount; j++) {
            char buf[256] = {0};

            cl::CLDeviceInfo info;
            size_t size;
            // Get device name
            clCall(clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buf), buf, &size));

            info.name = std::string(buf, size);

            int cores = 0;
            clCall(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cores), &cores, NULL));

            info.cores = cores;

            size_t maxWorkingGroupSize = 0;
            clCall(clGetDeviceInfo(devices[j], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkingGroupSize, NULL));
            info.maxWorkingGroupSize = maxWorkingGroupSize;

            cl_ulong mem;
            clCall(clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL));

            info.mem = (uint64_t)mem;
            info.id = devices[j];
            deviceList.push_back(info);
        }

        delete[] devices;
    }

    delete[] platforms;

    return deviceList;
}
