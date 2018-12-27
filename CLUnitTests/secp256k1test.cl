
#define SECTION_ADD 0
#define SECTION_MULTIPLY 1
#define SECTION_INVERSE 2

typedef struct {
    int section;
}CLErrorInfo;


bool addTest()
{
    unsigned int x[8] = { 0xa4aea9b8, 0x6fe248f5, 0x1fc74965, 0xe9493264, 0x4e2dff0c, 0x009f7c9c, 0x832fa59b, 0x3361f837 };
    unsigned int y[8] = { 0x537c13eb, 0xec1bf1f8, 0x7c25b4cf, 0xa57084ac, 0x245a823e, 0x624d20ee, 0x066cffaf, 0x4a0538f3 };
    unsigned int z[8] = { 0xf82abda4, 0x5bfe3aed, 0x9becfe35, 0x8eb9b710, 0x7288814a, 0x62ec9d8a, 0x899ca54a, 0x7d67312a };
    unsigned int k[8];

    addModP(x, y, k);

    return equal(z, k);
}

bool multiplyTest()
{
    unsigned int x[8] = { 0xa4aea9b8, 0x6fe248f5, 0x1fc74965, 0xe9493264, 0x4e2dff0c, 0x009f7c9c, 0x832fa59b, 0x3361f837 };
    unsigned int y[8] = { 0x537c13eb, 0xec1bf1f8, 0x7c25b4cf, 0xa57084ac, 0x245a823e, 0x624d20ee, 0x066cffaf, 0x4a0538f3 };
    unsigned int z[8] = { 0x4e0ce587, 0x119dd71e, 0x797c3d8c, 0x218d8631, 0x2535962b, 0xd61c1d6d, 0x01f40664, 0x3367edcb };
    unsigned int k[8];

    mulModP(x, y, k);

    return equal(z, k);
}

bool inverseTest()
{
    unsigned int x[8] = { 0xa4aea9b8, 0x6fe248f5, 0x1fc74965, 0xe9493264, 0x4e2dff0c, 0x009f7c9c, 0x832fa59b, 0x3361f837 };
    unsigned int k[8];
    unsigned int z[8] = { 0x22a595b4, 0x5a57167e, 0x1b9426be, 0x2c9b13e1, 0x8ca6f21c, 0x1765b9a9, 0xb378bbb3, 0x9a7f38e5 };

    for(int i = 0; i < 8; i++) {
        k[i] = x[i];
    }

    invModP(k);

    return equal(z, k);
}

void addError(__global CLErrorInfo *errInfo, __global unsigned int *numErrors, int section)
{
    unsigned int idx = atomic_add(numErrors, 1);
    
    CLErrorInfo info;
    info.section = section;

    errInfo[idx] = info;

}

__kernel void secp256k1_test(__global CLErrorInfo *errInfo, __global unsigned int *numErrors)
{
    if(!addTest()) {
        addError(errInfo, numErrors, SECTION_ADD);
    }

    if(!multiplyTest()) {
        addError(errInfo, numErrors, SECTION_MULTIPLY);
    }

    if(!inverseTest()) {
        addError(errInfo, numErrors, SECTION_INVERSE);
    }
}