
#include <vector>
#include <iostream>
#include "clutil.h"
#include "clContext.h"

#define SECTION_ADD 0
#define SECTION_MULTIPLY 1
#define SECTION_INVERSE 2

std::string _sections[] = {
    "Addition",
    "Multiplication",
    "Inverse" };

typedef struct {
    int section;
}CLErrorInfo;

extern char _secp256k1_test_cl[];

int runTest(cl_device_id deviceId)
{
    bool pass = true;

    cl::CLContext ctx(deviceId);
    cl::CLProgram prog(ctx, _secp256k1_test_cl);
    cl::CLKernel k(prog, "secp256k1_test");

    cl_mem devNumErrors = ctx.malloc(sizeof(unsigned int));
    cl_mem devErrors = ctx.malloc(sizeof(CLErrorInfo) * 1000);

    std::cout << "Running test kernel..." << std::endl;

    k.call(1, 1, devErrors, devNumErrors);

    unsigned int numErrors = 0;
    std::vector<CLErrorInfo> errors;

    ctx.copyDeviceToHost(devNumErrors, &numErrors, sizeof(unsigned int));

    std::cout << numErrors << " errors" << std::endl;

    if(numErrors > 0) {
        errors.resize(numErrors);

        ctx.copyDeviceToHost(devErrors, errors.data(), sizeof(CLErrorInfo) * numErrors);

        for(int i = 0; i < numErrors; i++) {
            std::cout << _sections[errors[i].section] << " test failed" << std::endl;
        }

        pass = false;
    }

    ctx.free(devNumErrors);
    ctx.free(devErrors);

    return numErrors;
}

int main(int argc, char **argv)
{
    std::vector<cl::CLDeviceInfo> devices = cl::getDevices();

    std::cout << "Found " << devices.size() << " devices" << std::endl;

    if(devices.size() == 0) {
        std::cout << "No OpenCL devices found" << std::endl;
        return 0;
    }

    int numErrors = 0;

    for(int i = 0; i < devices.size(); i++) {
        try {
            std::cout << "Testing device " << devices[i].name << std::endl;
            numErrors += runTest(devices[i].id);
        }
        catch(cl::CLException ex) {
            std::cout << "Error " << ex.msg << std::endl;
        }
    }

    std::cout << std::endl;

    if(!numErrors) {
        std::cout << "PASS" << std::endl;
    }
    else {
        std::cout << "FAIL" << std::endl;
    }

    return 0;
}