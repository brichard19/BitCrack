#ifndef _SECP256K1_CL
#define _SECP256K1_CL

typedef ulong uint64_t;

typedef struct {
    uint v[8];
}uint256_t;


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

__constant unsigned int _INFINITY[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF
};

void printBigInt(const unsigned int x[8])
{
    printf("%.8x %.8x %.8x %.8x %.8x %.8x %.8x %.8x\n",
        x[0], x[1], x[2], x[3],
        x[4], x[5], x[6], x[7]);
}

// Add with carry
unsigned int addc(unsigned int a, unsigned int b, unsigned int *carry)
{
    unsigned int sum = a + *carry;

    unsigned int c1 = (sum < a) ? 1 : 0;

    sum = sum + b;
    
    unsigned int c2 = (sum < b) ? 1 : 0;

    *carry = c1 | c2;

    return sum;
}

// Subtract with borrow
unsigned int subc(unsigned int a, unsigned int b, unsigned int *borrow)
{
    unsigned int diff = a - *borrow;

    *borrow = (diff > a) ? 1 : 0;

    unsigned int diff2 = diff - b;

    *borrow |= (diff2 > diff) ? 1 : 0;

    return diff2;
}

#ifdef DEVICE_VENDOR_INTEL

// Intel devices have a mul_hi bug
unsigned int mul_hi977(unsigned int x)
{
    unsigned int high = x >> 16;
    unsigned int low = x & 0xffff;

    return (((low * 977) >> 16) + (high * 977)) >> 16;
}

// 32 x 32 multiply-add
void madd977(unsigned int *high, unsigned int *low, unsigned int a, unsigned int c)
{
    *low = a * 977;
    unsigned int tmp = *low + c;
    unsigned int carry = tmp < *low ? 1 : 0;
    *low = tmp;
    *high = mul_hi977(a) + carry;
}

#else

// 32 x 32 multiply-add
void madd977(unsigned int *high, unsigned int *low, unsigned int a, unsigned int c)
{
    *low = a * 977;
    unsigned int tmp = *low + c;
    unsigned int carry = tmp < *low ? 1 : 0;
    *low = tmp;
    *high = mad_hi(a, (unsigned int)977, carry);
}

#endif

// 32 x 32 multiply-add
void madd(unsigned int *high, unsigned int *low, unsigned int a, unsigned int b, unsigned int c)
{
    *low = a * b;
    unsigned int tmp = *low + c;
    unsigned int carry = tmp < *low ? 1 : 0;
    *low = tmp;
    *high = mad_hi(a, b, carry);
}

void mull(unsigned int *high, unsigned int *low, unsigned int a, unsigned int b)
{
    *low = a * b;
    *high = mul_hi(a, b);
}


uint256_t sub256k(uint256_t a, uint256_t b, unsigned int* borrow_ptr)
{
    unsigned int borrow = 0;
    uint256_t c;

    for(int i = 7; i >= 0; i--) {
        c.v[i] = subc(a.v[i], b.v[i], &borrow);
    }

    *borrow_ptr = borrow;

    return c;
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

    return true;
}

void multiply256(const unsigned int x[8], const unsigned int y[8], unsigned int out_high[8], unsigned int out_low[8])
{
    unsigned int z[16];

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

    for(int i = 0; i < 8; i++) {
        out_high[i] = z[i];
        out_low[i] = z[8 + i];
    }
}


unsigned int add256(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    unsigned int carry = 0;

    for(int i = 7; i >= 0; i--) {
        c[i] = addc(a[i], b[i], &carry);
    }

    return carry;
}

uint256_t add256k(uint256_t a, uint256_t b, unsigned int* carry_ptr)
{
    uint256_t c;
    unsigned int carry = 0;

    for(int i = 7; i >= 0; i--) {
        c.v[i] = addc(a.v[i], b.v[i], &carry);
    }

    *carry_ptr = carry;

    return c;
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

bool isInfinity256k(const uint256_t x)
{
    bool isf = true;

    for(int i = 0; i < 8; i++) {
        if(x.v[i] != 0xffffffff) {
            isf = false;
        }
    }

    return isf;
}

bool equal(const unsigned int a[8], const unsigned int b[8])
{
    for(int i = 0; i < 8; i++) {
        if(a[i] != b[i]) {
            return false;
        }
    }

    return true;
}

bool equal256k(uint256_t a, uint256_t b)
{
    for(int i = 0; i < 8; i++) {
        if(a.v[i] != b.v[i]) {
            return false;
        }
    }

    return true;
}

inline uint256_t readInt256(__global const uint256_t* ara, int idx)
{
    return ara[idx];
}

/*
 * Read least-significant word
 */
unsigned int readLSW(__global const unsigned int *ara, int idx)
{
    return ara[idx * 8 + 7];
}

unsigned int readLSW256k(__global const uint256_t* ara, int idx)
{
    return ara[idx].v[7];
}

unsigned int readWord256k(__global const uint256_t* ara, int idx, int word)
{
    return ara[idx].v[word];
}

unsigned int addP(const unsigned int a[8], unsigned int c[8])
{
    unsigned int carry = 0;

    for(int i = 7; i >= 0; i--) {
        c[i] = addc(a[i], _P[i], &carry);
    }

    return carry;
}

unsigned int subP(const unsigned int a[8], unsigned int c[8])
{
    unsigned int borrow = 0;
    for(int i = 7; i >= 0; i--) {
        c[i] = subc(a[i], _P[i], &borrow);
    }

    return borrow;
}

/**
 * Subtraction mod p
 */
uint256_t subModP256k(uint256_t a, uint256_t b)
{
    unsigned int borrow = 0;
    uint256_t c = sub256k(a, b, &borrow);
    if(borrow) {
        addP(c.v, c.v);
    }

    return c;
}


uint256_t addModP256k(uint256_t a, uint256_t b)
{
    unsigned int carry = 0;

    uint256_t c = add256k(a, b, &carry);

    bool gt = false;
    for(int i = 0; i < 8; i++) {
        if(c.v[i] > _P[i]) {
            gt = true;
            break;
        } else if(c.v[i] < _P[i]) {
            break;
        }
    }

    if(carry || gt) {
        subP(c.v, c.v);
    }

    return c;
}


void mulModP(const unsigned int a[8], const unsigned int b[8], unsigned int product_low[8])
{
    unsigned int high[8];

    unsigned int hWord = 0;
    unsigned int carry = 0;

    // 256 x 256 multiply
    multiply256(a, b, high, product_low);

    // Add 2^32 * high to the low 256 bits (shift left 1 word and add)
    // Affects product[14] to product[6]
    for(int i = 6; i >= 0; i--) {
        product_low[i] = addc(product_low[i], high[i + 1], &carry);
    }
    unsigned int product7 = addc(high[0], 0, &carry);
    unsigned int product6 = carry;

    carry = 0;

    // Multiply high by 977 and add to low
    // Affects product[15] to product[5]
    for(int i = 7; i >= 0; i--) {
        unsigned int t = 0;
        madd977(&hWord, &t, high[i], hWord);
        product_low[i] = addc(product_low[i], t, &carry);
    }
    product7 = addc(product7, hWord, &carry);
    product6 = addc(product6, 0, &carry);

    // Multiply high 2 words by 2^32 and add to low
    // Affects product[14] to product[7]
    carry = 0;
    high[7] = product7;
    high[6] = product6;

    product7 = 0;
    product6 = 0;

    product_low[6] = addc(product_low[6], high[7], &carry);
    product_low[5] = addc(product_low[5], high[6], &carry);

    // Propagate the carry
    for(int i = 4; i >= 0; i--) {
        product_low[i] = addc(product_low[i], 0, &carry);
    }
    product7 = carry;

    // Multiply top 2 words by 977 and add to low
    // Affects product[15] to product[7]
    carry = 0;
    hWord = 0;
    unsigned int t = 0;
    madd977(&hWord, &t, high[7], hWord);
    product_low[7] = addc(product_low[7], t, &carry);
    madd977(&hWord, &t, high[6], hWord);
    product_low[6] = addc(product_low[6], t, &carry);
    product_low[5] = addc(product_low[5], hWord, &carry);

    // Propagate carry
    for(int i = 4; i >= 0; i--) {
        product_low[i] = addc(product_low[i], 0, &carry);
    }
    product7 = carry;

    // Reduce if >= P
    if(product7 || greaterThanEqualToP(product_low)) {
        subP(product_low, product_low);
    }
}

uint256_t mulModP256k(uint256_t a, uint256_t b)
{
    uint256_t c;

    mulModP(a.v, b.v, c.v);

    return c;
}


uint256_t squareModP256k(uint256_t a)
{
    uint256_t b;
    mulModP(a.v, a.v, b.v);

    return b;
}


/**
 * Multiplicative inverse mod P using Fermat's method of x^(p-2) mod p and addition chains
 */
uint256_t invModP256k(uint256_t value)
{
    uint256_t x = value;


    //unsigned int y[8] = { 0, 0, 0, 0, 0, 0, 0, 1 };
    uint256_t y = {{0, 0, 0, 0, 0, 0, 0, 1}};

    // 0xd - 1101
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    //y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);

    // 0x2 - 0010
    //y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    //y = mulModP256k(x, y);
    x = squareModP256k(x);
    //y = mulModP256k(x, y);
    x = squareModP256k(x);

    // 0xc = 0x1100
    //y = mulModP256k(x, y);
    x = squareModP256k(x);
    //y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);


    // 0xfffff
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);


    // 0xe - 1110
    //y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    y = mulModP256k(x, y);
    x = squareModP256k(x);
    // 0xfffffffffffffffffffffffffffffffffffffffffffffffffffffff
    for(int i = 0; i < 219; i++) {
        y = mulModP256k(x, y);
        x = squareModP256k(x);
    }
    y = mulModP256k(x, y);

    return y;
}


void beginBatchAdd256k(uint256_t px, uint256_t x, __global uint256_t* chain, int i, int batchIdx, uint256_t* inverse)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    // x = Gx - x
    uint256_t t = subModP256k(px, x);


    // Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
    // c[2] = diff2 * diff1 * diff0, etc
    *inverse = mulModP256k(*inverse, t);

    chain[batchIdx * dim + gid] = *inverse;
}


void beginBatchAddWithDouble256k(uint256_t px, uint256_t py, __global uint256_t* xPtr, __global uint256_t* chain, int i, int batchIdx, uint256_t* inverse)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    uint256_t x = xPtr[i];

    if(equal256k(px, x)) {
        x = addModP256k(py, py);
    } else {
        // x = Gx - x
        x = subModP256k(px, x);
    }

    // Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
    // c[2] = diff2 * diff1 * diff0, etc
    *inverse = mulModP256k(x, *inverse);

    chain[batchIdx * dim + gid] = *inverse;
}


void completeBatchAddWithDouble256k(
    uint256_t px,
    uint256_t py,
    __global const uint256_t* xPtr,
    __global const uint256_t* yPtr,
    int i,
    int batchIdx,
    __global uint256_t* chain,
    uint256_t* inverse,
    uint256_t* newX,
    uint256_t* newY)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);
    uint256_t s;
    uint256_t x;
    uint256_t y;

    x = xPtr[i];
    y = yPtr[i];

    if(batchIdx >= 1) {

        uint256_t c;

        c = chain[(batchIdx - 1) * dim + gid];
        s = mulModP256k(*inverse, c);

        uint256_t diff;
        if(equal256k(px, x)) {
            diff = addModP256k(py, py);
        } else {
            diff = subModP256k(px, x);
        }

        *inverse = mulModP256k(diff, *inverse);
    } else {
        s = *inverse;
    }


    if(equal256k(px, x)) {
        // currently s = 1 / 2y

        uint256_t x2;
        uint256_t tx2;
        uint256_t x3;

        // 3x^2
        x2 = mulModP256k(x, x);
        tx2 = addModP256k(x2, x2);
        tx2 = addModP256k(x2, tx2);

        // s = 3x^2 * 1/2y
        s = mulModP256k(tx2, s);

        // s^2
        uint256_t s2;
        s2 = mulModP256k(s, s);

        // Rx = s^2 - 2px
        *newX = subModP256k(s2, x);
        *newX = subModP256k(*newX, x);

        // Ry = s(px - rx) - py
        uint256_t k;
        k = subModP256k(px, *newX);
        *newY = mulModP256k(s, k);
        *newY = subModP256k(*newY, py);
    } else {

        uint256_t rise;
        rise = subModP256k(py, y);

        s = mulModP256k(rise, s);

        // Rx = s^2 - Gx - Qx
        uint256_t s2;
        s2 = mulModP256k(s, s);

        *newX = subModP256k(s2, px);
        *newX = subModP256k(*newX, x);

        // Ry = s(px - rx) - py
        uint256_t k;
        k = subModP256k(px, *newX);
        *newY = mulModP256k(s, k);
        *newY = subModP256k(*newY, py);
    }
}


void completeBatchAdd256k(
    uint256_t px,
    uint256_t py,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    int i,
    int batchIdx,
    __global uint256_t* chain,
    uint256_t* inverse,
    uint256_t* newX,
    uint256_t* newY)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    uint256_t s;
    uint256_t x;

    x = xPtr[i];

    if(batchIdx >= 1) {
        uint256_t c;

        c = chain[(batchIdx - 1) * dim + gid];
        s = mulModP256k(*inverse, c);

        uint256_t diff;
        diff = subModP256k(px, x);
        *inverse = mulModP256k(diff, *inverse);
    } else {
        s = *inverse;
    }

    uint256_t y;
    y = yPtr[i];

    uint256_t rise;
    rise = subModP256k(py, y);

    s = mulModP256k(rise, s);

    // Rx = s^2 - Gx - Qx
    uint256_t s2;
    s2 = mulModP256k(s, s);

    *newX = subModP256k(s2, px);
    *newX = subModP256k(*newX, x);

    // Ry = s(px - rx) - py
    uint256_t k;
    k = subModP256k(px, *newX);
    *newY = mulModP256k(s, k);
    *newY = subModP256k(*newY, py);
}


uint256_t doBatchInverse256k(uint256_t x)
{
    return invModP256k(x);
}

#endif
