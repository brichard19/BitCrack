#ifndef _SECP256K1_CL
#define _SECP256K1_CL

typedef ulong uint64_t;

/**
 Prime modulus 2^256 - 2^32 - 977
 */
__constant unsigned int _P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

__constant unsigned int _P_MINUS1[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

/**
 Base point X
 */
__constant unsigned int _GX[8] = {
    0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};

/**
 Base point Y
 */
__constant unsigned int _GY[8] = {
    0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};


/**
 * Group order
 */
__constant unsigned int _N[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};



// Add with carry
#define addc(sum, a, b, carry)\
{\
    uint64_t sum64 = (uint64_t)a + b + carry;\
    sum = (unsigned int)sum64;\
    carry = (unsigned int)(sum64 >> 32) & 1;\
}

// Subtract with borrow
#define subc(diff, a, b, borrow)\
{\
    uint64_t diff64 = (uint64_t)a - b - borrow;\
    diff = (unsigned int)diff64;\
    borrow = (unsigned int)((diff64 >> 32) & 1);\
}

// 32 x 32 > 64 multiply
#define mul(high, low, a, b)\
{\
    uint64_t prod64 = (uint64_t)a * b;\
    low = (unsigned int)prod64;\
    high = (unsigned int)(prod64 >> 32);\
}

// 32 x 32 multiply-add
#define madd(high, low, a, b, c)\
{\
    uint64_t mul64 = (uint64_t)a * b + c;\
    low = (unsigned int)mul64;\
    high = (unsigned int)(mul64 >> 32);\
}

unsigned int sub256(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    int borrow = 0;
    for(int i = 7; i >= 0; i--) {
        subc(c[i], a[i], b[i], borrow);
    }

    return borrow;
}

bool greaterThanEqualToP(const unsigned int a[8])
{
    for(int i = 0; i < 8; i++) {
        if(a[i] > _P_MINUS1[i]) {
            return true;
        } else if(a[i] < _P_MINUS1[i]) {
            return false;
        }
    }

    return false;
}

void multiply256(const unsigned int x[8], const unsigned int y[8], unsigned int z[16])
{
    unsigned int high = 0;

    // First round, overwrite z
    for(int j = 7; j >= 0; j--) {

        uint64_t product = (uint64_t)x[7] * y[j];

        product = product + high;

        z[7 + j + 1] = (unsigned int)product;
        high = (unsigned int)(product >> 32);
    }
    z[7] = high;

    for(int i = 6; i >= 0; i--) {

        high = 0;

        for(int j = 7; j >= 0; j--) {

            uint64_t product = (uint64_t)x[i] * y[j];

            product = product + z[i + j + 1] + high;

            z[i + j + 1] = (unsigned int)product;

            high = product >> 32;
        }

        z[i] = high;
    }    
}

unsigned int add256(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    int carry = 0;

    for(int i = 7; i >= 0; i--) {
        addc(c[i], a[i], b[i], carry);
    }

    return carry;
}



bool isInfinity(const unsigned int x[8])
{
    bool isf = true;

    for(int i = 0; i < 8; i++) {
        if(x[i] != 0xffffffff) {
            isf = false;
        }
    }

    return isf;
}

void copyBigInt(const unsigned int src[8], unsigned int dest[8])
{
    for(int i = 0; i < 8; i++) {
        dest[i] = src[i];
    }
}

bool equal(const unsigned int *a, const unsigned int *b)
{
    bool eq = true;

    for(int i = 0; i < 8; i++) {
        eq &= (a[i] == b[i]);
    }

    return eq;
}

/**
 * Reads an 8-word big integer from device memory
 */
void readInt(__global const unsigned int *ara, int idx, unsigned int x[8])
{
    size_t totalThreads = get_global_size(0);

    size_t base = idx * totalThreads * 8;

    size_t threadId = get_local_size(0) * get_group_id(0) + get_local_id(0);

    for(int i = 0; i < 8; i++) {
        x[i] = ara[base + threadId * 8 + i];
    }
}

/*
 * Read least-significant word
 */
unsigned int readLSW(__global const unsigned int *ara, int idx)
{
    size_t totalThreads = get_global_size(0);

    size_t base = idx * totalThreads * 8;

    size_t threadId = get_local_size(0) * get_group_id(0) + get_local_id(0);

    return ara[base + threadId * 8 + 7];
}

/**
 * Writes an 8-word big integer to device memory
 */
void writeInt(__global unsigned int *ara, int idx, const unsigned int x[8])
{
    size_t totalThreads = get_global_size(0);

    size_t base = idx * totalThreads * 8;

    size_t threadId = get_local_size(0) * get_group_id(0) + get_local_id(0);

    for(int i = 0; i < 8; i++) {
        ara[base + threadId * 8 + i] = x[i];
    }
}

unsigned int addP(const unsigned int a[8], unsigned int c[8])
{
    int carry = 0;

    for(int i = 7; i >= 0; i--) {
        addc(c[i], a[i], _P[i], carry);
    }

    return carry;
}

unsigned int subP(const unsigned int a[8], unsigned int c[8])
{
    int borrow = 0;
    for(int i = 7; i >= 0; i--) {
        subc(c[i], a[i], _P[i], borrow);
    }

    return borrow;
}

/**
 * Subtraction mod p
 */
void subModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    if(sub256(a, b, c)) {
        addP(c, c);
    }
}


void addModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    int carry = 0;

    carry = add256(a, b, c);

    bool gt = false;
    for(int i = 0; i < 8; i++) {
        if(c[i] > _P[i]) {
            gt = true;
            break;
        } else if(c[i] < _P[i]) {
            break;
        }
    }

    if(carry || gt) {
        subP(c, c);
    }
}


void mulModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    unsigned int product[16];
    unsigned int hWord = 0;
    unsigned int carry = 0;

    // 256 x 256 multiply
    multiply256(a, b, product);

    // Copy the high 256 bits
    unsigned int high[8];

    for(int i = 0; i < 8; i++) {
        high[i] = product[i];
    }

    for(int i = 0; i < 8; i++) {
        product[i] = 0;
    }

    // Add 2^32 * high to the low 256 bits (shift left 1 word and add)
    // Affects product[14] to product[6]
    for(int i = 7; i >= 0; i--) {
        addc(product[i + 7], product[i + 7], high[i], carry);
    }
    addc(product[6], product[6], 0, carry);

    carry = 0;

    // Multiply high by 977 and add to low
    // Affects product[15] to product[5]
    for(int i = 7; i >= 0; i--) {
        unsigned int t;
        madd(hWord, t, high[i], 977, hWord);
        addc(product[8 + i], product[8 + i], t, carry);
    }
    addc(product[7], product[7], hWord, carry);
    addc(product[6], 0, 0, carry);

    // Multiply high 2 words by 2^32 and add to low
    // Affects product[14] to product[7]
    carry = 0;
    high[7] = product[7];
    high[6] = product[6];

    product[7] = 0;
    product[6] = 0;

    addc(product[14], product[14], high[7], carry);
    addc(product[13], product[13], high[6], carry);

    // Propagate the carry
    for(int i = 12; i >= 7; i--) {
        addc(product[i], product[i], 0, carry);
    }

    // Multiply top 2 words by 977 and add to low
    // Affects product[15] to product[7]
    carry = 0;
    hWord = 0;
    unsigned int t;
    madd(hWord, t, high[7], 977, hWord);
    addc(product[15], product[15], t, carry);
    madd(hWord, t, high[6], 977, hWord);
    addc(product[14], product[14], t, carry);
    addc(product[13], product[13], hWord, carry);

    // Propagate carry
    for(int i = 12; i >= 7; i--) {
        addc(product[i], product[i], 0, carry);
    }

    // Reduce if >= P
    if(product[7] || greaterThanEqualToP(&product[8])) {
        subP(&product[8], &product[8]);
    }

    for(int i = 0; i < 8; i++) {
        c[i] = product[8 + i];
    }
}

/**
 * Multiply mod P
 * c = a * c
 */
void mulModP_d(const unsigned int a[8], unsigned int c[8])
{
    unsigned int tmp[8];
    mulModP(a, c, tmp);

    copyBigInt(tmp, c);
}

/**
 * Square mod P
 * b = a * a
 */
void squareModP(const unsigned int a[8], unsigned int b[8])
{
    mulModP(a, a, b);
}

/**
 * Square mod P
 * x = x * x
 */
void squareModP_d(unsigned int x[8])
{
    unsigned int tmp[8];
    squareModP(x, tmp);
    copyBigInt(tmp, x);
}



/**
 * Multiplicative inverse mod P using Fermat's method of x^(p-2) mod p and addition chains
 */
void invModP(unsigned int value[8])
{
    unsigned int x[8];

    copyBigInt(value, x);

    unsigned int y[8] = {0, 0, 0, 0, 0, 0, 0, 1};

    // 0xd - 1101
    mulModP_d(x, y);
    squareModP_d(x);
    //mulModP_d(x, y);
    squareModP_d(x);
    mulModP_d(x, y);
    squareModP_d(x);
    mulModP_d(x, y);
    squareModP_d(x);


    // 0x2 - 0010
    //mulModP_d(x, y);
    squareModP_d(x);
    mulModP_d(x, y);
    squareModP_d(x);
    //mulModP_d(x, y);
    squareModP_d(x);
    //mulModP_d(x, y);
    squareModP_d(x);

    // 0xc = 0x1100
    //mulModP_d(x, y);
    squareModP_d(x);
    //mulModP_d(x, y);
    squareModP_d(x);
    mulModP_d(x, y);
    squareModP_d(x);
    mulModP_d(x, y);
    squareModP_d(x);

    // 0xfffff
    for(int i = 0; i < 20; i++) {
        mulModP_d(x, y);
        squareModP_d(x);
    }

    // 0xe - 1110
    //mulModP_d(x, y);
    squareModP_d(x);
    mulModP_d(x, y);
    squareModP_d(x);
    mulModP_d(x, y);
    squareModP_d(x);
    mulModP_d(x, y);
    squareModP_d(x);

    // 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffff
    for(int i = 0; i < 219; i++) {
        mulModP_d(x, y);
        squareModP_d(x);
    }
    mulModP_d(x, y);

    copyBigInt(y, value);
}


void beginBatchAdd(const unsigned int *px, const unsigned int *x, __global unsigned int *chain, int i, int batchIdx, unsigned int inverse[8])
{
    // x = Gx - x
    unsigned int t[8];
    subModP(px, x, t);

    // Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
    // c[2] = diff2 * diff1 * diff0, etc
    mulModP_d(t, inverse);

    writeInt(chain, batchIdx, inverse);
}


void beginBatchAddWithDouble(const unsigned int *px, const unsigned int *py, __global unsigned int *xPtr, __global unsigned int *chain, int i, int batchIdx, unsigned int inverse[8])
{
    unsigned int x[8];
    readInt(xPtr, i, x);

    if(equal(px, x)) {
        addModP(py, py, x);
    } else {
        // x = Gx - x
        subModP(px, x, x);
    }

    // Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
    // c[2] = diff2 * diff1 * diff0, etc
    mulModP_d(x, inverse);

    writeInt(chain, batchIdx, inverse);
}

void completeBatchAddWithDouble(
    const unsigned int *px,
    const unsigned int *py,
    __global const unsigned int *xPtr,
    __global const unsigned int *yPtr,
    int i,
    int batchIdx,
    __global unsigned int *chain,
    unsigned int *inverse,
    unsigned int newX[8],
    unsigned int newY[8])
{
    unsigned int s[8];
    unsigned int x[8];
    unsigned int y[8];

    readInt(xPtr, i, x);
    readInt(yPtr, i, y);

    if(batchIdx >= 1) {
        unsigned int c[8];

        readInt(chain, batchIdx - 1, c);

        mulModP(inverse, c, s);

        unsigned int diff[8];
        if(equal(px, x)) {
            addModP(py, py, diff);
        } else {
            subModP(px, x, diff);
        }

        mulModP_d(diff, inverse);
    } else {
        copyBigInt(inverse, s);
    }


    if(equal(px, x)) {
        // currently s = 1 / 2y

        unsigned int x2[8];
        unsigned int tx2[8];

        // 3x^2
        mulModP(x, x, x2);
        addModP(x2, x2, tx2);
        addModP(x2, tx2, tx2);


        // s = 3x^2 * 1/2y
        mulModP_d(tx2, s);

        // s^2
        unsigned int s2[8];
        mulModP(s, s, s2);

        // Rx = s^2 - 2px
        subModP(s2, x, newX);
        subModP(newX, x, newX);

        // Ry = s(px - rx) - py
        unsigned int k[8];
        subModP(px, newX, k);
        mulModP(s, k, newY);
        subModP(newY, py, newY);

    } else {

        unsigned int rise[8];
        subModP(py, y, rise);

        mulModP_d(rise, s);

        // Rx = s^2 - Gx - Qx
        unsigned int s2[8];
        mulModP(s, s, s2);

        subModP(s2, px, newX);
        subModP(newX, x, newX);

        // Ry = s(px - rx) - py
        unsigned int k[8];
        subModP(px, newX, k);
        mulModP(s, k, newY);
        subModP(newY, py, newY);
    }
}

void completeBatchAdd(
    const unsigned int *px,
    const unsigned int *py,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr,
    int i,
    int batchIdx,
    __global unsigned int *chain,
    unsigned int *inverse,
    unsigned int newX[8],
    unsigned int newY[8])
{
    unsigned int s[8];
    unsigned int x[8];

    readInt(xPtr, i, x);

    if(batchIdx >= 1) {
        unsigned int c[8];

        readInt(chain, batchIdx - 1, c);
        mulModP(inverse, c, s);

        unsigned int diff[8];
        subModP(px, x, diff);
        mulModP_d(diff, inverse);
    } else {
        copyBigInt(inverse, s);
    }

    unsigned int y[8];
    readInt(yPtr, i, y);

    unsigned int rise[8];
    subModP(py, y, rise);

    mulModP_d(rise, s);

    // Rx = s^2 - Gx - Qx
    unsigned int s2[8];
    mulModP(s, s, s2);
    subModP(s2, px, newX);
    subModP(newX, x, newX);

    // Ry = s(px - rx) - py
    unsigned int k[8];
    subModP(px, newX, k);
    mulModP(s, k, newY);
    subModP(newY, py, newY);
}


void doBatchInverse(unsigned int inverse[8])
{
    invModP(inverse);
}

#endif
