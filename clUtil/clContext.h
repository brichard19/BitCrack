#ifndef _CL_CONTEXT_H
#define _CL_CONTEXT_H

#include <string>
#include <CL/cl.h>
#include "clutil.h"

namespace cl {

class CLContext {

private:
    cl_device_id _device;
    cl_context _ctx;
    cl_command_queue _queue;

public:
    CLContext(cl_device_id device);
    ~CLContext();

    cl_device_id getDevice();

    cl_context getContext();

    cl_command_queue getQueue();


    cl_mem malloc(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE);
    void free(cl_mem mem);
    void memset(cl_mem mem, int value, size_t size);
    void copyHostToDevice(const void *hostPtr, cl_mem devicePtr, size_t size);
    void copyHostToDevice(const void *hostPtr, cl_mem devicePtr, size_t offset, size_t size);
    void copyDeviceToHost(cl_mem devicePtr, void *hostPtr, size_t size);
    std::string getDeviceName();
    uint64_t getGlobalMemorySize();
};

class CLProgram {

private:
    cl_program _prog;
    CLContext &_ctx;
    std::string _buildLog;

    std::string loadSource(std::string src);

public:
    CLProgram(CLContext &ctx, std::string src);
    CLProgram(CLContext &ctx, const char *src);

    ~CLProgram();

    cl_program getProgram();

    CLContext& getContext();

    std::string getBuildLog();

};


class CLKernel {

private:
    CLProgram &_prog;
    std::string _entry;
    cl_kernel _kernel;

public:
    CLKernel(CLProgram &prog, std::string entry);
    size_t getWorkGroupSize();

    ~CLKernel();

    template<typename T1>
    void call(size_t blocks, size_t threads, T1 arg1)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(arg1), &arg1));

        clCall(clEnqueueNDRangeKernel(_prog.getContext().getQueue(), _kernel, 1, NULL, &blocks, &threads, 0, NULL, NULL));
        clFinish(_prog.getContext().getQueue());
    }

    template<typename T1, typename T2>
    void call(size_t blocks, size_t threads, T1 arg1, T2 arg2)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(arg1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(arg2), &arg2));

        clCall(clEnqueueNDRangeKernel(_prog.getContext().getQueue(), _kernel, 1, NULL, &blocks, &threads, 0, NULL, NULL));
        clFinish(_prog.getContext().getQueue());
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    void call(size_t blocks, size_t threads, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(arg1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(arg2), &arg2));
        clCall(clSetKernelArg(_kernel, 2, sizeof(arg3), &arg3));
        clCall(clSetKernelArg(_kernel, 3, sizeof(arg4), &arg4));
        clCall(clSetKernelArg(_kernel, 4, sizeof(arg5), &arg5));

        clCall(clEnqueueNDRangeKernel(_prog.getContext().getQueue(), _kernel, 1, NULL, &blocks, &threads, 0, NULL, NULL));
        clFinish(_prog.getContext().getQueue());
    }


    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
    void call(size_t blocks, size_t threads, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(T1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(T2), &arg2));
        clCall(clSetKernelArg(_kernel, 2, sizeof(T3), &arg3));
        clCall(clSetKernelArg(_kernel, 3, sizeof(T4), &arg4));
        clCall(clSetKernelArg(_kernel, 4, sizeof(T5), &arg5));
        clCall(clSetKernelArg(_kernel, 5, sizeof(T6), &arg6));
        clCall(clSetKernelArg(_kernel, 6, sizeof(T7), &arg7));
        clCall(clSetKernelArg(_kernel, 7, sizeof(T8), &arg8));

        size_t totalThreads = blocks * threads;

        clCall(clEnqueueNDRangeKernel(_prog.getContext().getQueue(), _kernel, 1, NULL, &totalThreads, &threads, 0, NULL, NULL));
        clCall(clFinish(_prog.getContext().getQueue()));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
    void call(size_t blocks, size_t threads, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(T1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(T2), &arg2));
        clCall(clSetKernelArg(_kernel, 2, sizeof(T3), &arg3));
        clCall(clSetKernelArg(_kernel, 3, sizeof(T4), &arg4));
        clCall(clSetKernelArg(_kernel, 4, sizeof(T5), &arg5));
        clCall(clSetKernelArg(_kernel, 5, sizeof(T6), &arg6));
        clCall(clSetKernelArg(_kernel, 6, sizeof(T7), &arg7));
        clCall(clSetKernelArg(_kernel, 7, sizeof(T8), &arg8));
        clCall(clSetKernelArg(_kernel, 8, sizeof(T9), &arg9));
        clCall(clSetKernelArg(_kernel, 9, sizeof(T10), &arg10));

        size_t totalThreads = blocks * threads;

        clCall(clEnqueueNDRangeKernel(_prog.getContext().getQueue(), _kernel, 1, NULL, &totalThreads, &threads, 0, NULL, NULL));
        clCall(clFinish(_prog.getContext().getQueue()));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
    void call(size_t blocks, size_t threads, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(T1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(T2), &arg2));
        clCall(clSetKernelArg(_kernel, 2, sizeof(T3), &arg3));
        clCall(clSetKernelArg(_kernel, 3, sizeof(T4), &arg4));
        clCall(clSetKernelArg(_kernel, 4, sizeof(T5), &arg5));
        clCall(clSetKernelArg(_kernel, 5, sizeof(T6), &arg6));
        clCall(clSetKernelArg(_kernel, 6, sizeof(T7), &arg7));
        clCall(clSetKernelArg(_kernel, 7, sizeof(T8), &arg8));
        clCall(clSetKernelArg(_kernel, 8, sizeof(T9), &arg9));
        clCall(clSetKernelArg(_kernel, 9, sizeof(T10), &arg10));
        clCall(clSetKernelArg(_kernel, 10, sizeof(T11), &arg11));

        size_t totalThreads = blocks * threads;

        clCall(clEnqueueNDRangeKernel(_prog.getContext().getQueue(), _kernel, 1, NULL, &totalThreads, &threads, 0, NULL, NULL));
        clCall(clFinish(_prog.getContext().getQueue()));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
    void call(size_t blocks, size_t threads, T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(T1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(T2), &arg2));
        clCall(clSetKernelArg(_kernel, 2, sizeof(T3), &arg3));
        clCall(clSetKernelArg(_kernel, 3, sizeof(T4), &arg4));
        clCall(clSetKernelArg(_kernel, 4, sizeof(T5), &arg5));
        clCall(clSetKernelArg(_kernel, 5, sizeof(T6), &arg6));
        clCall(clSetKernelArg(_kernel, 6, sizeof(T7), &arg7));
        clCall(clSetKernelArg(_kernel, 7, sizeof(T8), &arg8));
        clCall(clSetKernelArg(_kernel, 8, sizeof(T9), &arg9));
        clCall(clSetKernelArg(_kernel, 9, sizeof(T10), &arg10));
        clCall(clSetKernelArg(_kernel, 10, sizeof(T11), &arg11));
        clCall(clSetKernelArg(_kernel, 11, sizeof(T12), &arg12));

        size_t totalThreads = blocks * threads;

        clCall(clEnqueueNDRangeKernel(_prog.getContext().getQueue(), _kernel, 1, NULL, &totalThreads, &threads, 0, NULL, NULL));
        clCall(clFinish(_prog.getContext().getQueue()));
    }
};



}
#endif