#include <fstream>
#include <string>
#include <sstream>

#include "clContext.h"



cl::CLContext::CLContext(cl_device_id device)
{
    _device = device;

    cl_int err;
    _ctx = clCreateContext(0, 1, &_device, NULL, NULL, &err);
    clCall(err);

    _queue = clCreateCommandQueue(_ctx, _device, 0, &err);
    clCall(err);
}

cl::CLContext::~CLContext()
{
    clReleaseCommandQueue(_queue);
    clReleaseContext(_ctx);
}

cl_device_id cl::CLContext::getDevice()
{
    return _device;
}

cl_command_queue cl::CLContext::getQueue()
{
    return _queue;
}

cl_context cl::CLContext::getContext()
{
    return _ctx;
}

cl_mem cl::CLContext::malloc(size_t size, cl_mem_flags flags)
{
    cl_int err = 0;
    cl_mem ptr = clCreateBuffer(_ctx, flags, size, NULL, &err);
    clCall(err);

    return ptr;
}

void cl::CLContext::free(cl_mem mem)
{
    clReleaseMemObject(mem);
}

void cl::CLContext::copyHostToDevice(const void *hostPtr, cl_mem devicePtr, size_t size)
{
   clCall(clEnqueueWriteBuffer(_queue, devicePtr, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL));
    
}

void cl::CLContext::copyHostToDevice(const void *hostPtr, cl_mem devicePtr, size_t offset, size_t size)
{
    clCall(clEnqueueWriteBuffer(_queue, devicePtr, CL_TRUE, offset, size, hostPtr, 0, NULL, NULL));
}

void cl::CLContext::copyDeviceToHost(cl_mem devicePtr, void *hostPtr, size_t size)
{
    clCall(clEnqueueReadBuffer(_queue, devicePtr, CL_TRUE, 0, size, hostPtr, 0, NULL, NULL));
}

void cl::CLContext::memset(cl_mem devicePtr, int value, size_t size)
{
    clCall(clEnqueueFillBuffer(_queue, devicePtr, &value, sizeof(int), 0, size, NULL, NULL, NULL));
}

cl::CLProgram::CLProgram(cl::CLContext &ctx, std::string srcFile) : _ctx(ctx)
{
    std::string src = loadSource(srcFile);
    const char *ptr = src.c_str();
    size_t len = src.length();
    cl_int err;

    _prog = clCreateProgramWithSource(ctx.getContext(), 1, &ptr, &len, &err);
    clCall(err);

    err = clBuildProgram(_prog, 0, NULL, NULL, NULL, NULL);

    if(err == CL_BUILD_PROGRAM_FAILURE) {
        size_t logSize;
        clGetProgramBuildInfo(_prog, ctx.getDevice(), CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

        char *log = new char[logSize];
        clGetProgramBuildInfo(_prog, ctx.getDevice(), CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

        _buildLog = std::string(log, logSize);
        delete[] log;

        throw CLException(err, _buildLog);
    }
    clCall(err);
}

cl::CLProgram::CLProgram(cl::CLContext &ctx, const char *src) : _ctx(ctx)
{
    size_t len = strlen(src);
    cl_int err;

    _prog = clCreateProgramWithSource(ctx.getContext(), 1, &src, &len, &err);
    clCall(err);

    err = clBuildProgram(_prog, 0, NULL, NULL, NULL, NULL);

    if(err == CL_BUILD_PROGRAM_FAILURE) {
        size_t logSize;
        clGetProgramBuildInfo(_prog, ctx.getDevice(), CL_PROGRAM_BUILD_LOG, 0, NULL, &logSize);

        char *log = new char[logSize];
        clGetProgramBuildInfo(_prog, ctx.getDevice(), CL_PROGRAM_BUILD_LOG, logSize, log, NULL);

        _buildLog = std::string(log, logSize);
        delete[] log;

        throw CLException(err, _buildLog);
    }
    clCall(err);
}

std::string cl::CLProgram::loadSource(std::string srcFile)
{
    std::ifstream f(srcFile);
    if(!f.good()) {
        throw CLException(CL_BUILD_PROGRAM_FAILURE, "'" + srcFile + "' not found");
    }

    std::stringstream buf;
    buf << f.rdbuf();

    return buf.str();
}

cl_program cl::CLProgram::getProgram()
{
    return _prog;
}

cl::CLContext& cl::CLProgram::getContext()
{
    return _ctx;
}

uint64_t cl::CLContext::getGlobalMemorySize()
{
    cl_ulong mem;
    clCall(clGetDeviceInfo(_device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(mem), &mem, NULL));

    return mem;
}

std::string cl::CLContext::getDeviceName()
{
    char name[128] = {0};

    clCall(clGetDeviceInfo(_device, CL_DEVICE_NAME, sizeof(name), name, NULL));

    return std::string(name);
}

cl::CLProgram::~CLProgram()
{
    clReleaseProgram(_prog);
}


cl::CLKernel::CLKernel(cl::CLProgram &prog, std::string entry) : _prog(prog)
{
    _entry = entry;
    const char *ptr = entry.c_str();
    cl_int err;
    _kernel = clCreateKernel(_prog.getProgram(), ptr, &err);
    clCall(err);
}

size_t cl::CLKernel::getWorkGroupSize()
{
    size_t size = 0;

    cl_int err = clGetKernelWorkGroupInfo(_kernel, _prog.getContext().getDevice(), CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &size, NULL);

    clCall(err);

    return size;
}

cl::CLKernel::~CLKernel()
{
    clReleaseKernel(_kernel);
}