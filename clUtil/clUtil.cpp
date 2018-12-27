#include<CL/cl.h>
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

    cl_platform_id* platforms = new cl_platform_id[platformCount];

    clGetPlatformIDs(platformCount, platforms, NULL);

    for(cl_uint i = 0; i < platformCount; i++) {
        
        cl_uint deviceCount;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceCount);

        if(deviceCount == 0) {
            continue;
        }

        cl_device_id* devices = new cl_device_id[deviceCount];
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, deviceCount, devices, NULL);

        for(cl_uint j = 0; j < deviceCount; j++) {
            char buf[256] = {0};

            cl::CLDeviceInfo info;
            size_t size;
            // Get device name
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buf), buf, &size);

            info.name = std::string(buf, size);

            int cores = 0;
            clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cores), &cores, NULL);

            info.cores = cores;

            cl_ulong mem;
            clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL);

            info.mem = (uint64_t)mem;
            info.id = devices[j];
            deviceList.push_back(info);
        }

        delete devices;
    }

    delete platforms;

    return deviceList;
}