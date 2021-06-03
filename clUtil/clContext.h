#ifndef CL_CONTEXT_H
#define CL_CONTEXT_H

#include <string>
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
    void memset(cl_mem mem, unsigned char value, size_t size);
    void copyHostToDevice(const void *hostPtr, cl_mem devicePtr, size_t size);
    void copyHostToDevice(const void *hostPtr, cl_mem devicePtr, size_t offset, size_t size);
    void copyDeviceToHost(cl_mem devicePtr, void *hostPtr, size_t size);
    void copyBuffer(cl_mem src_buffer, size_t src_offset, cl_mem dst_buffer, size_t dst_offset, size_t size);
    std::string getDeviceName();
    std::string getDeviceVendor();
    int get_mp_count();
    int get_max_block_size();

    uint64_t getGlobalMemorySize();
};

class CLProgram {

private:
    cl_program _prog;
    CLContext &_ctx;
    std::string _buildLog;

    std::string loadSource(std::string src);

public:
    CLProgram(CLContext &ctx, std::string src, std::string options = "");
    CLProgram(CLContext &ctx, const char *src, std::string options = "");

    ~CLProgram();

    cl_program getProgram();

    CLContext& getContext();
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
    void set_args(T1 arg1)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(arg1), &arg1));
    }

    template<typename T1, typename T2>
    void set_args(T1 arg1, T2 arg2)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(arg1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(arg2), &arg2));
    }

    template<typename T1, typename T2, typename T3>
    void set_args(T1 arg1, T2 arg2, T3 arg3)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(arg1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(arg2), &arg2));
        clCall(clSetKernelArg(_kernel, 2, sizeof(arg3), &arg3));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5>
    void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(arg1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(arg2), &arg2));
        clCall(clSetKernelArg(_kernel, 2, sizeof(arg3), &arg3));
        clCall(clSetKernelArg(_kernel, 3, sizeof(arg4), &arg4));
        clCall(clSetKernelArg(_kernel, 4, sizeof(arg5), &arg5));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6>
    void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(arg1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(arg2), &arg2));
        clCall(clSetKernelArg(_kernel, 2, sizeof(arg3), &arg3));
        clCall(clSetKernelArg(_kernel, 3, sizeof(arg4), &arg4));
        clCall(clSetKernelArg(_kernel, 4, sizeof(arg5), &arg5));
        clCall(clSetKernelArg(_kernel, 5, sizeof(arg6), &arg6));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8>
    void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8)
    {
        clCall(clSetKernelArg(_kernel, 0, sizeof(T1), &arg1));
        clCall(clSetKernelArg(_kernel, 1, sizeof(T2), &arg2));
        clCall(clSetKernelArg(_kernel, 2, sizeof(T3), &arg3));
        clCall(clSetKernelArg(_kernel, 3, sizeof(T4), &arg4));
        clCall(clSetKernelArg(_kernel, 4, sizeof(T5), &arg5));
        clCall(clSetKernelArg(_kernel, 5, sizeof(T6), &arg6));
        clCall(clSetKernelArg(_kernel, 6, sizeof(T7), &arg7));
        clCall(clSetKernelArg(_kernel, 7, sizeof(T8), &arg8));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10>
    void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10)
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
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11>
    void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11)
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
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8, typename T9, typename T10, typename T11, typename T12>
    void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12)
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
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8,
        typename T9, typename T10, typename T11, typename T12, typename T13, typename T14>
        void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12,
            T13 arg13, T14 arg14)
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
        clCall(clSetKernelArg(_kernel, 12, sizeof(T13), &arg13));
        clCall(clSetKernelArg(_kernel, 13, sizeof(T14), &arg14));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8,
        typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15>
        void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12,
            T13 arg13, T14 arg14, T15 arg15)
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
        clCall(clSetKernelArg(_kernel, 12, sizeof(T13), &arg13));
        clCall(clSetKernelArg(_kernel, 13, sizeof(T14), &arg14));
        clCall(clSetKernelArg(_kernel, 14, sizeof(T15), &arg15));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8,
             typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16>
    void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12,
        T13 arg13, T14 arg14, T15 arg15, T16 arg16)
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
        clCall(clSetKernelArg(_kernel, 12, sizeof(T13), &arg13));
        clCall(clSetKernelArg(_kernel, 13, sizeof(T14), &arg14));
        clCall(clSetKernelArg(_kernel, 14, sizeof(T15), &arg15));
        clCall(clSetKernelArg(_kernel, 15, sizeof(T16), &arg16));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8,
        typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16,
        typename T17>
        void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12,
            T13 arg13, T14 arg14, T15 arg15, T16 arg16, T17 arg17)
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
        clCall(clSetKernelArg(_kernel, 12, sizeof(T13), &arg13));
        clCall(clSetKernelArg(_kernel, 13, sizeof(T14), &arg14));
        clCall(clSetKernelArg(_kernel, 14, sizeof(T15), &arg15));
        clCall(clSetKernelArg(_kernel, 15, sizeof(T16), &arg16));
        clCall(clSetKernelArg(_kernel, 16, sizeof(T17), &arg17));
    }

    template<typename T1, typename T2, typename T3, typename T4, typename T5, typename T6, typename T7, typename T8,
        typename T9, typename T10, typename T11, typename T12, typename T13, typename T14, typename T15, typename T16,
        typename T17, typename T18>
        void set_args(T1 arg1, T2 arg2, T3 arg3, T4 arg4, T5 arg5, T6 arg6, T7 arg7, T8 arg8, T9 arg9, T10 arg10, T11 arg11, T12 arg12,
            T13 arg13, T14 arg14, T15 arg15, T16 arg16, T17 arg17, T18 arg18)
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
        clCall(clSetKernelArg(_kernel, 12, sizeof(T13), &arg13));
        clCall(clSetKernelArg(_kernel, 13, sizeof(T14), &arg14));
        clCall(clSetKernelArg(_kernel, 14, sizeof(T15), &arg15));
        clCall(clSetKernelArg(_kernel, 15, sizeof(T16), &arg16));
        clCall(clSetKernelArg(_kernel, 16, sizeof(T17), &arg17));
        clCall(clSetKernelArg(_kernel, 17, sizeof(T18), &arg18));
    }

    void call(size_t blocks, size_t threads)
    {
        size_t totalThreads = blocks * threads;
        clCall(clEnqueueNDRangeKernel(_prog.getContext().getQueue(), _kernel, 1, NULL, &totalThreads, &threads, 0, NULL, NULL));
        clCall(clFinish(_prog.getContext().getQueue()));
    }
};


}
#endif
