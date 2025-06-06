#ifndef _PTX_H
#define _PTX_H

#include<cuda_runtime.h>

#define madc_hi(dest, a, x, b) asm volatile("madc.hi.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_hi_cc(dest, a, x, b) asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define mad_hi_cc(dest, a, x, b) asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))

#define mad_lo_cc(dest, a, x, b) asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_lo(dest, a, x, b) asm volatile("madc.lo.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x), "r"(b))
#define madc_lo_cc(dest, a, x, b) asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;\n\t" : "=r"(dest) : "r"(a), "r"(x),"r"(b))

#define addc(dest, a, b) asm volatile("addc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define add_cc(dest, a, b) asm volatile("add.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define addc_cc(dest, a, b) asm volatile("addc.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))

#define sub_cc(dest, a, b) asm volatile("sub.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define subc_cc(dest, a, b) asm volatile("subc.cc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))
#define subc(dest, a, b) asm volatile("subc.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))

#define set_eq(dest,a,b) asm volatile("set.eq.u32.u32 %0, %1, %2;\n\t" : "=r"(dest) : "r"(a), "r"(b))

#define lsbpos(x) (__ffs((x)))


__device__ __forceinline__ unsigned int endian(unsigned int x)
{
	return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

#endif