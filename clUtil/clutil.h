#ifndef CL_UTIL_H
#define CL_UTIL_H

#ifdef __APPLE__
#define CL_SILENCE_DEPRECATION
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <string>
#include <vector>

namespace cl {
    std::string getOpenCLErrorName(cl_int errorCode);
    std::string getOpenCLErrorDescription(cl_int errorCode);

    typedef struct {
        cl_device_id id;
        int cores;
        uint64_t mem;
        std::string name;
        size_t maxWorkingGroupSize;
    }CLDeviceInfo;

    class CLException {
    public:
        int error;
        std::string msg;
        std::string description;

        CLException(cl_int errorCode)
        {
            this->error = errorCode;
            this->msg = getOpenCLErrorName(errorCode);
            this->description = getOpenCLErrorDescription(errorCode);
        }

        CLException(cl_int errorCode, std::string pMsg)
        {
            this->error = errorCode;
            this->msg = pMsg;
            this->description = getOpenCLErrorDescription(errorCode);
        }

        CLException(cl_int errorCode, std::string pMsg, std::string pDescription)
        {
            this->error = errorCode;
            this->msg = pMsg;
            this->description = pDescription;
        }
    };

    CLDeviceInfo getDeviceInfo(int device);

    std::vector<CLDeviceInfo> getDevices();

    void clCall(cl_int err);

}

#endif
