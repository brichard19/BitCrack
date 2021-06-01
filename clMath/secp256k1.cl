#ifndef SECP256K1_CL
#define SECP256K1_CL

typedef struct uint256_t {
    unsigned int v[8];
} uint256_t;

/**
 * Base point X
 */
__constant unsigned int GX[8] = {
    0x79BE667E, 0xF9DCBBAC, 0x55A06295, 0xCE870B07, 0x029BFCDB, 0x2DCE28D9, 0x59F2815B, 0x16F81798
};

/**
 * Base point Y
 */
__constant unsigned int GY[8] = {
    0x483ADA77, 0x26A3C465, 0x5DA4FBFC, 0x0E1108A8, 0xFD17B448, 0xA6855419, 0x9C47D08F, 0xFB10D4B8
};

/**
 * Group order
 */
__constant unsigned int N[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xBAAEDCE6, 0xAF48A03B, 0xBFD25E8C, 0xD0364141
};

/**
 * Prime modulus 2^256 - 2^32 - 977
 */
__constant unsigned int P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

#ifdef DEVICE_VENDOR_INTEL
// Intel devices have a mul_hi bug
unsigned int mul_hi977(unsigned int x)
{
    unsigned int high = x >> 16;
    unsigned int low = x & 0xffff;

    return (((low * 977) >> 16) + (high * 977)) >> 16;
}

// 32 x 32 multiply-add
void madd977(unsigned int *high, unsigned int *low, unsigned int *a, unsigned int *c)
{
    *low = *a * 977;
    unsigned int tmp = *low + *c;
    unsigned int carry = tmp < *low ? 1 : 0;
    *low = tmp;
    *high = mul_hi977(*a) + carry;
}
#else

void madd977(unsigned int *high, unsigned int *low, unsigned int *a, unsigned int *c)
{
    *low = *a * 977;
    unsigned int tmp = *low + *c;
    unsigned int carry = tmp < *low ? 1 : 0;
    *low = tmp;
    *high = mad_hi(*a, (unsigned int)977, carry);
}

#endif

// Add with carry
#define addc(a, b, sum, carry, tmp)      \
    sum = (a) + (carry);                 \
    tmp = ((sum) < (a)) * 1;             \
    sum = (sum) + (b);                   \
    carry = (tmp) | (((sum) < (b)) * 1);

// subtract with borrow
#define subc(a, b, diff, borrow, tmp)    \
    tmp = (a) - (borrow);                \
    borrow = ((tmp) > (a)) * 1;          \
    diff = (tmp) - (b);                  \
    borrow |= ((diff) > (tmp)) ? 1 : 0;

#define add256k(a, b, c, carry, tmp)    \
    addc(a[7], b[7], c[7], carry, tmp); \
    addc(a[6], b[6], c[6], carry, tmp); \
    addc(a[5], b[5], c[5], carry, tmp); \
    addc(a[4], b[4], c[4], carry, tmp); \
    addc(a[3], b[3], c[3], carry, tmp); \
    addc(a[2], b[2], c[2], carry, tmp); \
    addc(a[1], b[1], c[1], carry, tmp); \
    addc(a[0], b[0], c[0], carry, tmp);

#define sub256k( a, b, c, borrow, tmp)   \
    subc(a[7], b[7], c[7], borrow, tmp); \
    subc(a[6], b[6], c[6], borrow, tmp); \
    subc(a[5], b[5], c[5], borrow, tmp); \
    subc(a[4], b[4], c[4], borrow, tmp); \
    subc(a[3], b[3], c[3], borrow, tmp); \
    subc(a[2], b[2], c[2], borrow, tmp); \
    subc(a[1], b[1], c[1], borrow, tmp); \
    subc(a[0], b[0], c[0], borrow, tmp);

#define isInfinity256k(a)        \
    (                           \
        (a[0] == 0xffffffff) && \
        (a[1] == 0xffffffff) && \
        (a[2] == 0xffffffff) && \
        (a[3] == 0xffffffff) && \
        (a[4] == 0xffffffff) && \
        (a[5] == 0xffffffff) && \
        (a[6] == 0xffffffff) && \
        (a[7] == 0xffffffff)    \
    )

#define greaterOrEqualToP(a)    \
    (a[6] >= P[6] || a[7] >= P[7])

#define equal256k(a, b)   \
    (                     \
        (a[0] == b[0]) && \
        (a[1] == b[1]) && \
        (a[2] == b[2]) && \
        (a[3] == b[3]) && \
        (a[4] == b[4]) && \
        (a[5] == b[5]) && \
        (a[6] == b[6]) && \
        (a[7] == b[7])    \
    )

void multiply256(const unsigned int x[8], const unsigned int y[8], unsigned int out_high[8], unsigned int out_low[8])
{
    unsigned long product;

    // First round, overwrite z
    product = (unsigned long)x[7] * y[7];
    out_low[7] = (unsigned int)product;
    
    product = (unsigned long)x[7] * y[6] + (unsigned int)(product >> 32);
    out_low[6] = (unsigned int)product;
    
    product = (unsigned long)x[7] * y[5] + (unsigned int)(product >> 32);
    out_low[5] = (unsigned int)product;
    
    product = (unsigned long)x[7] * y[4] + (unsigned int)(product >> 32);
    out_low[4] = (unsigned int)product;
    
    product = (unsigned long)x[7] * y[3] + (unsigned int)(product >> 32);
    out_low[3] = (unsigned int)product;
    
    product = (unsigned long)x[7] * y[2] + (unsigned int)(product >> 32);
    out_low[2] = (unsigned int)product;
        
    product = (unsigned long)x[7] * y[1] + (unsigned int)(product >> 32);
    out_low[1] = (unsigned int)product;
        
    product = (unsigned long)x[7] * y[0] + (unsigned int)(product >> 32);
    out_low[0] = (unsigned int)product;
    out_high[7] = (unsigned int)(product >> 32);

    product = (unsigned long)x[6] * y[7] + out_low[6];
    out_low[6] = (unsigned int)product;

    /** round6 */
    product = (unsigned long)x[6] * y[6] + out_low[5] + (product >> 32);
    out_low[5] = (unsigned int)product;

    product = (unsigned long)x[6] * y[5] + out_low[4] + (product >> 32);
    out_low[4] = (unsigned int)product;

    product = (unsigned long)x[6] * y[4] + out_low[3] + (product >> 32);
    out_low[3] = (unsigned int)product;

    product = (unsigned long)x[6] * y[3] + out_low[2] + (product >> 32);
    out_low[2] = (unsigned int)product;

    product = (unsigned long)x[6] * y[2] + out_low[1] + (product >> 32);
    out_low[1] = (unsigned int)product;
    
    product = (unsigned long)x[6] * y[1] + out_low[0] + (product >> 32);
    out_low[0] = (unsigned int)product;
    
    product = (unsigned long)x[6] * y[0] + out_high[7] + (product >> 32);
    out_high[7] = (unsigned int)product;
    out_high[6] = product >> 32;

    /** round 5 */
    product = (unsigned long)x[5] * y[7] + out_low[5];
    out_low[5] = (unsigned int)product;

    product = (unsigned long)x[5] * y[6] + out_low[4] + (product >> 32);
    out_low[4] = (unsigned int)product;

    product = (unsigned long)x[5] * y[5] + out_low[3] + (product >> 32);
    out_low[3] = (unsigned int)product;

    product = (unsigned long)x[5] * y[4] + out_low[2] + (product >> 32);
    out_low[2] = (unsigned int)product;

    product = (unsigned long)x[5] * y[3] + out_low[1] + (product >> 32);
    out_low[1] = (unsigned int)product;

    product = (unsigned long)x[5] * y[2] + out_low[0] + (product >> 32);
    out_low[0] = (unsigned int)product;
    
    product = (unsigned long)x[5] * y[1] + out_high[7] + (product >> 32);
    out_high[7] = (unsigned int)product;
    
    product = (unsigned long)x[5] * y[0] + out_high[6] + (product >> 32);
    out_high[6] = (unsigned int)product;
    out_high[5] = product >> 32;

    /** round 4 */
    product = (unsigned long)x[4] * y[7] + out_low[4];
    out_low[4] = (unsigned int)product;

    product = (unsigned long)x[4] * y[6] + out_low[3] + (product >> 32);
    out_low[3] = (unsigned int)product;

    product = (unsigned long)x[4] * y[5] + out_low[2] + (product >> 32);
    out_low[2] = (unsigned int)product;

    product = (unsigned long)x[4] * y[4] + out_low[1] + (product >> 32);
    out_low[1] = (unsigned int)product;

    product = (unsigned long)x[4] * y[3] + out_low[0] + (product >> 32);
    out_low[0] = (unsigned int)product;

    product = (unsigned long)x[4] * y[2] + out_high[7] + (product >> 32);
    out_high[7] = (unsigned int)product;
    
    product = (unsigned long)x[4] * y[1] + out_high[6] + (product >> 32);
    out_high[6] = (unsigned int)product;
    
    product = (unsigned long)x[4] * y[0] + out_high[5] + (product >> 32);
    out_high[5] = (unsigned int)product;
    out_high[4] = product >> 32;

    /** round 3 */
    product = (unsigned long)x[3] * y[7] + out_low[3];
    out_low[3] = (unsigned int)product;

    product = (unsigned long)x[3] * y[6] + out_low[2] + (product >> 32);
    out_low[2] = (unsigned int)product;

    product = (unsigned long)x[3] * y[5] + out_low[1] + (product >> 32);
    out_low[1] = (unsigned int)product;

    product = (unsigned long)x[3] * y[4] + out_low[0] + (product >> 32);
    out_low[0] = (unsigned int)product;

    product = (unsigned long)x[3] * y[3] + out_high[7] + (product >> 32);
    out_high[7] = (unsigned int)product;

    product = (unsigned long)x[3] * y[2] + out_high[6] + (product >> 32);
    out_high[6] = (unsigned int)product;
    
    product = (unsigned long)x[3] * y[1] + out_high[5] + (product >> 32);
    out_high[5] = (unsigned int)product;
    
    product = (unsigned long)x[3] * y[0] + out_high[4] + (product >> 32);
    out_high[4] = (unsigned int)product;
    out_high[3] = product >> 32;

    /** round 2 */
    product = (unsigned long)x[2] * y[7] + out_low[2];
    out_low[2] = (unsigned int)product;

    product = (unsigned long)x[2] * y[6] + out_low[1] + (product >> 32);
    out_low[1] = (unsigned int)product;

    product = (unsigned long)x[2] * y[5] + out_low[0] + (product >> 32);
    out_low[0] = (unsigned int)product;

    product = (unsigned long)x[2] * y[4] + out_high[7] + (product >> 32);
    out_high[7] = (unsigned int)product;

    product = (unsigned long)x[2] * y[3] + out_high[6] + (product >> 32);
    out_high[6] = (unsigned int)product;

    product = (unsigned long)x[2] * y[2] + out_high[5] + (product >> 32);
    out_high[5] = (unsigned int)product;
    
    product = (unsigned long)x[2] * y[1] + out_high[4] + (product >> 32);
    out_high[4] = (unsigned int)product;
    
    product = (unsigned long)x[2] * y[0] + out_high[3] + (product >> 32);
    out_high[3] = (unsigned int)product;
    out_high[2] = product >> 32;
    
    /** round 1 */
    product = (unsigned long)x[1] * y[7] + out_low[1];
    out_low[1] = (unsigned int)product;

    product = (unsigned long)x[1] * y[6] + out_low[0] + (product >> 32);
    out_low[0] = (unsigned int)product;

    product = (unsigned long)x[1] * y[5] + out_high[7] + (product >> 32);
    out_high[7] = (unsigned int)product;

    product = (unsigned long)x[1] * y[4] + out_high[6] + (product >> 32);
    out_high[6] = (unsigned int)product;

    product = (unsigned long)x[1] * y[3] + out_high[5] + (product >> 32);
    out_high[5] = (unsigned int)product;

    product = (unsigned long)x[1] * y[2] + out_high[4] + (product >> 32);
    out_high[4] = (unsigned int)product;
    
    product = (unsigned long)x[1] * y[1] + out_high[3] + (product >> 32);
    out_high[3] = (unsigned int)product;
    
    product = (unsigned long)x[1] * y[0] + out_high[2] + (product >> 32);
    out_high[2] = (unsigned int)product;
    out_high[1] = product >> 32;

    /** round 0 */
    product = (unsigned long)x[0] * y[7] + out_low[0];
    out_low[0] = (unsigned int)product;

    product = (unsigned long)x[0] * y[6] + out_high[7] + (product >> 32);
    out_high[7] = (unsigned int)product;

    product = (unsigned long)x[0] * y[5] + out_high[6] + (product >> 32);
    out_high[6] = (unsigned int)product;

    product = (unsigned long)x[0] * y[4] + out_high[5] + (product >> 32);
    out_high[5] = (unsigned int)product;

    product = (unsigned long)x[0] * y[3] + out_high[4] + (product >> 32);
    out_high[4] = (unsigned int)product;

    product = (unsigned long)x[0] * y[2] + out_high[3] + (product >> 32);
    out_high[3] = (unsigned int)product;
    
    product = (unsigned long)x[0] * y[1] + out_high[2] + (product >> 32);
    out_high[2] = (unsigned int)product;
    
    product = (unsigned long)x[0] * y[0] + out_high[1] + (product >> 32);
    out_high[1] = (unsigned int)product;
    out_high[0] = product >> 32;
}

void mulModP(unsigned int a[8], unsigned int b[8], unsigned int product_low[8])
{
    unsigned int high[8];
    unsigned int low[8];

    unsigned int hWord = 0;
    unsigned int carry = 0;
    unsigned int t = 0;
    unsigned int product6 = 0;
    unsigned int product7 = 0;
    unsigned int tmp;


    // 256 x 256 multiply
    multiply256(a, b, high, low);
    product_low[7] = low[7];
    product_low[6] = low[6];
    product_low[5] = low[5];
    product_low[4] = low[4];
    product_low[3] = low[3];
    product_low[2] = low[2];
    product_low[1] = low[1];
    product_low[0] = low[0];

    // Add 2^32 * high to the low 256 bits (shift left 1 word and add)
    // Affects product[14] to product[6]
    addc(product_low[6], high[7], product_low[6], carry, tmp);
    addc(product_low[5], high[6], product_low[5], carry, tmp);
    addc(product_low[4], high[5], product_low[4], carry, tmp);
    addc(product_low[3], high[4], product_low[3], carry, tmp);
    addc(product_low[2], high[3], product_low[2], carry, tmp);
    addc(product_low[1], high[2], product_low[1], carry, tmp);
    addc(product_low[0], high[1], product_low[0], carry, tmp);

    addc(high[0], 0, product7, carry, tmp);
    product6 = carry;

    carry = 0;

    // Multiply high by 977 and add to low
    // Affects product[15] to product[5]
    for(int i = 7; i >= 0; i--) {
        madd977(&hWord, &t, &high[i], &hWord);
        addc(product_low[i], t, product_low[i], carry, tmp);
        t = 0;
    }
    addc(product7, hWord, high[7], carry, tmp);
    addc(product6, 0, high[6], carry, tmp);

    // Multiply high 2 words by 2^32 and add to low
    // Affects product[14] to product[7]
    carry = 0;

    addc(product_low[6], high[7], product_low[6], carry, tmp);
    addc(product_low[5], high[6], product_low[5], carry, tmp);

    addc(product_low[4], 0, product_low[4], carry, tmp);
    addc(product_low[3], 0, product_low[3], carry, tmp);
    addc(product_low[2], 0, product_low[2], carry, tmp);
    addc(product_low[1], 0, product_low[1], carry, tmp);
    addc(product_low[0], 0, product_low[0], carry, tmp);

    // Multiply top 2 words by 977 and add to low
    // Affects product[15] to product[7]
    carry = 0;
    hWord = 0;
    madd977(&hWord, &t, &high[7], &hWord);
    addc(product_low[7], t, product_low[7], carry, tmp);
    madd977(&hWord, &t, &high[6], &hWord);
    addc(product_low[6], t,  product_low[6], carry, tmp);
    addc(product_low[5], hWord,  product_low[5], carry, tmp);
    // Propagate carry
    addc(product_low[4], 0, product_low[4], carry, tmp);
    addc(product_low[3], 0, product_low[3], carry, tmp);
    addc(product_low[2], 0, product_low[2], carry, tmp);
    addc(product_low[1], 0, product_low[1], carry, tmp);
    addc(product_low[0], 0, product_low[0], carry, tmp);

    // Reduce if >= P
    if(carry || greaterOrEqualToP(product_low)) {
        carry = 0;
        sub256k(product_low, P, product_low, carry, tmp);
    }
}

/**
 * Subtraction mod p
 */
void subModP256k(unsigned int a[8], unsigned int b[8], unsigned int c[8])
{
    unsigned int borrow = 0;
    unsigned int tmp;
    
    sub256k(a, b, c, borrow, tmp);
    
    if (borrow) {
        borrow = 0;
        add256k(c, P, c, borrow, tmp);
    }
}

/**
 * Multiplicative inverse mod P using Fermat's method of x^(p-2) mod p and addition chains
 */
void invModP256k(unsigned int x[8])
{
    unsigned int y[8] = {0, 0, 0, 0, 0, 0, 0, 1};

    mulModP(x, y, y);
    mulModP(x, x, x);
    mulModP(x, x, x);
    mulModP(x, y, y);
    mulModP(x, x, x);
    mulModP(x, y, y);
    mulModP(x, x, x);
    mulModP(x, x, x);
    mulModP(x, y, y);

    for(int i = 0; i < 5; i++) {
        mulModP(x, x, x);
    }

    for(int i = 0; i < 22; i++) {
        mulModP(x, y, y);
        mulModP(x, x, x);
    }

    mulModP(x, x, x);

    for(int i = 0; i < 222; i++) {
        mulModP(x, y, y);
        mulModP(x, x, x);
    }

    mulModP(x, y, x);
}

void addModP256k(unsigned int a[8], unsigned int b[8], unsigned int c[8])
{
    unsigned int borrow = 0;
    unsigned int carry = 0;
    unsigned int tmp = 0;

    add256k(a, b, c, carry, tmp);

    if(carry) { sub256k(c, P, c, borrow, tmp); }

    else if(c[0] > P[0]) { sub256k(c, P, c, borrow, tmp); } 
    else if(c[0] < P[0]) {  }

    else if(c[1] > P[1]) { sub256k(c, P, c, borrow, tmp); } 
    else if(c[1] < P[1]) {  }

    else if(c[2] > P[2]) { sub256k(c, P, c, borrow, tmp); } 
    else if(c[2] < P[2]) {  }
    
    else if(c[3] > P[3]) { sub256k(c, P, c, borrow, tmp); } 
    else if(c[3] < P[3]) {  }
    
    else if(c[4] > P[4]) { sub256k(c, P, c, borrow, tmp); } 
    else if(c[4] < P[4]) {  }
    
    else if(c[5] > P[5]) { sub256k(c, P, c, borrow, tmp); } 
    else if(c[5] < P[5]) {  }
    
    else if(c[6] > P[6]) { sub256k(c, P, c, borrow, tmp); } 
    else if(c[6] < P[6]) {  }

    else if(c[7] > P[7]) { sub256k(c, P, c, borrow, tmp); } 
}

void doBatchInverse256k(unsigned int x[8])
{
    invModP256k(x);
}

void beginBatchAdd256k(uint256_t px, uint256_t x, __global uint256_t* chain, int i, int batchIdx, uint256_t* inverse)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    unsigned int t[8];

    // x = Gx - x
    subModP256k(px.v, x.v, t);


    // Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
    // c[2] = diff2 * diff1 * diff0, etc
    mulModP(inverse->v, t, inverse->v);

    chain[batchIdx * dim + gid] = *inverse;
}

void beginBatchAddWithDouble256k(uint256_t px, uint256_t py, __global uint256_t* xPtr, __global uint256_t* chain, int i, int batchIdx, uint256_t* inverse)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);
    uint256_t x = xPtr[i];

    if(equal256k(px.v, x.v)) {
        addModP256k(py.v,py.v, x.v);
    } else {
        // x = Gx - x
        subModP256k(px.v, x.v, x.v);
    }

    // Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
    // c[2] = diff2 * diff1 * diff0, etc
    mulModP(x.v, inverse->v, inverse->v);

    chain[batchIdx * dim + gid] = *inverse;
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
    uint256_t x = xPtr[i];
    uint256_t y = yPtr[i];
	
    uint256_t s;
    unsigned int tmp[8];

    if(batchIdx != 0) {
        uint256_t c;

        c = chain[(batchIdx - 1) * dim + gid];
        mulModP(inverse->v, c.v, s.v);

        subModP256k(px.v, x.v, tmp);
        mulModP(tmp, inverse->v, inverse->v);
    } else {
        s = *inverse;
    }

	subModP256k(py.v, y.v, tmp);

    mulModP(tmp, s.v, s.v);

    // Rx = s^2 - Gx - Qx
    mulModP(s.v, s.v, tmp);

    subModP256k(tmp, px.v, newX->v);
    subModP256k(newX->v, x.v, newX->v);

    // Ry = s(px - rx) - py
	subModP256k(px.v, newX->v, tmp);
    mulModP(s.v, tmp, newY->v);
    subModP256k(newY->v, py.v, newY->v);
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
        mulModP(inverse->v, c.v, s.v);

        uint256_t diff;
        if(equal256k(px.v, x.v)) {
            addModP256k(py.v, py.v, diff.v);
        } else {
            subModP256k(px.v, x.v, diff.v);
        }

        mulModP(diff.v, inverse->v, inverse->v);
    } else {
        s = *inverse;
    }


    if(equal256k(px.v, x.v)) {
        // currently s = 1 / 2y

        uint256_t x2;
        uint256_t tx2;

        // 3x^2
        mulModP(x.v, x.v, x2.v);
        addModP256k(x2.v, x2.v, tx2.v);
        addModP256k(x2.v, tx2.v, tx2.v);

        // s = 3x^2 * 1/2y
        mulModP(tx2.v, s.v, s.v);

        // s^2
        uint256_t s2;
        mulModP(s.v, s.v, s2.v);

        // Rx = s^2 - 2px
        subModP256k(s2.v, x.v, newX->v);
        subModP256k(newX->v, x.v, newX->v);

        // Ry = s(px - rx) - py
        uint256_t k;
				subModP256k(px.v, newX->v, k.v);
        mulModP(s.v, k.v, newY->v);
        subModP256k(newY->v, py.v,newY->v);
    } else {

        uint256_t rise;
        subModP256k(py.v, y.v, rise.v);

        mulModP(rise.v, s.v, s.v);

        // Rx = s^2 - Gx - Qx
        uint256_t s2;
        mulModP(s.v, s.v, s2.v);

        subModP256k(s2.v, px.v, newX->v);
        subModP256k(newX->v, x.v,newX->v);

        // Ry = s(px - rx) - py
        uint256_t k;
        subModP256k(px.v, newX->v, k.v);
        mulModP(s.v, k.v, newY->v);
        subModP256k(newY->v, py.v, newY->v);
    }
}

unsigned int readLSW256k(__global const uint256_t* ara, int idx)
{
    return ara[idx].v[7];
}

unsigned int readWord256k(__global const uint256_t* ara, int idx, int word)
{
    return ara[idx].v[word];
}

#endif
