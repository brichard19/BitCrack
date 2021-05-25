#ifndef SECP256K1_CL
#define SECP256K1_CL

typedef unsigned long uint64_t;

typedef struct uint256_t {
    unsigned int v[8];
} uint256_t;

/**
 Prime modulus 2^256 - 2^32 - 977
 */
__constant unsigned int P[8] = {
    0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
};

// Add with carry
void addc(unsigned int *a, unsigned int *b, unsigned int *carry, unsigned int *sum)
{
    *sum = *a + *carry;

    unsigned int c1 = (*sum < *a) * 1;

    *sum = *sum + *b;
    
    *carry = c1 | ((*sum < *b) * 1);
}

// Subtract with borrow
void subc(unsigned int *a, unsigned int *b, unsigned int *borrow, unsigned int *diff)
{
    unsigned int tmp = *a - *borrow;

    *borrow = (tmp > *a) * 1;

    *diff = tmp - *b;

    *borrow |= (*diff > tmp) ? 1 : 0;
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
void madd977(unsigned int *high, unsigned int *low, unsigned int *a, unsigned int *c)
{
    *low = *a * 977;
    unsigned int tmp = *low + *c;
    unsigned int carry = tmp < *low ? 1 : 0;
    *low = tmp;
    *high = mul_hi977(*a) + carry;
}

#else

// 32 x 32 multiply-add
void madd977(unsigned int *high, unsigned int *low, unsigned int *a, unsigned int *c)
{
    *low = *a * 977;
    unsigned int tmp = *low + *c;
    unsigned int carry = tmp < *low ? 1 : 0;
    *low = tmp;
    *high = mad_hi(*a, (unsigned int)977, carry);
}

#endif

uint256_t sub256k(uint256_t a, uint256_t b, unsigned int* borrow_ptr)
{
    unsigned int borrow = 0;
    uint256_t c;

    subc(&a.v[7], &b.v[7], &borrow, &c.v[7]);
    subc(&a.v[6], &b.v[6], &borrow, &c.v[6]);
    subc(&a.v[5], &b.v[5], &borrow, &c.v[5]);
    subc(&a.v[4], &b.v[4], &borrow, &c.v[4]);
    subc(&a.v[3], &b.v[3], &borrow, &c.v[3]);
    subc(&a.v[2], &b.v[2], &borrow, &c.v[2]);
    subc(&a.v[1], &b.v[1], &borrow, &c.v[1]);
    subc(&a.v[0], &b.v[0], &borrow, &c.v[0]);

    *borrow_ptr = borrow;

    return c;
}

bool greaterThanEqualToP(const unsigned int a[8])
{
    if(a[0] > P[0]) { return true; } 
    if(a[0] < P[0]) { return false; }

    if(a[1] > P[1]) { return true; } 
    if(a[1] < P[1]) { return false; }
    
    if(a[2] > P[2]) { return true; } 
    if(a[2] < P[2]) { return false; }
    
    if(a[3] > P[3]) { return true; } 
    if(a[3] < P[3]) { return false; }
    
    if(a[4] > P[4]) { return true; } 
    if(a[4] < P[4]) { return false; }
    
    if(a[5] > P[5]) { return true; } 
    if(a[5] < P[5]) { return false; }
    
    if(a[6] > P[6]) { return true; } 
    if(a[6] < P[6]) { return false; }
    
    if(a[7] > P[7]) { return true; } 
    if(a[7] < P[7]) { return false; }

    return true;
}

void multiply256(const unsigned int x[8], const unsigned int y[8], unsigned int out_high[8], unsigned int out_low[8])
{
    unsigned int z[16];
    unsigned int high = 0;
    uint64_t product = 0;

    // First round, overwrite z
    for(int j = 7; j >= 0; j--) {

        product = (uint64_t)x[7] * y[j] + high;

        z[7 + j + 1] = (unsigned int)product;
        high = (unsigned int)(product >> 32);
    }
    z[7] = high;

    for(int i = 6; i >= 0; i--) {

        high = 0;

        for(int j = 7; j >= 0; j--) {

            product = (uint64_t)x[i] * y[j] + z[i + j + 1] + high;

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

uint256_t add256k(uint256_t a, uint256_t b, unsigned int* carry_ptr)
{
    uint256_t c;
    unsigned int carry = 0;

    for(int i = 7; i >= 0; i--) {
        addc(&a.v[i], &b.v[i], &carry, &c.v[i]);
    }

    *carry_ptr = carry;

    return c;
}

bool isInfinity256k(const uint256_t *x)
{
    return (
        (x->v[0] == 0xffffffff) &&
        (x->v[1] == 0xffffffff) &&
        (x->v[2] == 0xffffffff) &&
        (x->v[3] == 0xffffffff) &&
        (x->v[4] == 0xffffffff) &&
        (x->v[5] == 0xffffffff) &&
        (x->v[6] == 0xffffffff) &&
        (x->v[7] == 0xffffffff)
    );
}

bool equal256k(uint256_t *a, uint256_t *b)
{
    return (
        (a->v[0] == b->v[0]) &&
        (a->v[1] == b->v[1]) &&
        (a->v[2] == b->v[2]) &&
        (a->v[3] == b->v[3]) &&
        (a->v[4] == b->v[4]) &&
        (a->v[5] == b->v[5]) &&
        (a->v[6] == b->v[6]) &&
        (a->v[7] == b->v[7])
    );
}

unsigned int readLSW256k(__global const uint256_t* ara, int idx)
{
    return ara[idx].v[7];
}

unsigned int readWord256k(__global const uint256_t* ara, int idx, int word)
{
    return ara[idx].v[word];
}

void addP(unsigned int a[8], unsigned int c[8])
{
    unsigned int carry = 0;
    unsigned int P[8] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
    };
    addc(&a[7], &P[7], &carry, &c[7]);
    addc(&a[6], &P[6], &carry, &c[6]);
    addc(&a[5], &P[5], &carry, &c[5]);
    addc(&a[4], &P[4], &carry, &c[4]);
    addc(&a[3], &P[3], &carry, &c[3]);
    addc(&a[2], &P[2], &carry, &c[2]);
    addc(&a[1], &P[1], &carry, &c[1]);
    addc(&a[0], &P[0], &carry, &c[0]);
}

void subP(unsigned int a[8], unsigned int c[8])
{
    unsigned int borrow = 0;
    unsigned int P[8] = {
        0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFE, 0xFFFFFC2F
    };
    subc(&a[7], &P[7], &borrow, &c[7]);
    subc(&a[6], &P[6], &borrow, &c[6]);
    subc(&a[5], &P[5], &borrow, &c[5]);
    subc(&a[4], &P[4], &borrow, &c[4]);
    subc(&a[3], &P[3], &borrow, &c[3]);
    subc(&a[2], &P[2], &borrow, &c[2]);
    subc(&a[1], &P[1], &borrow, &c[1]);
    subc(&a[0], &P[0], &borrow, &c[0]);
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

void addModP256k(uint256_t *a, uint256_t *b, uint256_t *cP)
{
    unsigned int carry = 0;

    uint256_t c = add256k(*a, *b, &carry);

    if(carry) { subP(c.v, c.v); *cP = c; }

    else if(c.v[0] > P[0]) { subP(c.v, c.v); *cP = c; } 
    else if(c.v[0] < P[0]) { *cP = c; }

    else if(c.v[1] > P[1]) { subP(c.v, c.v); *cP = c; } 
    else if(c.v[1] < P[1]) { *cP = c; }

    else if(c.v[2] > P[2]) { subP(c.v, c.v); *cP = c; } 
    else if(c.v[2] < P[2]) { *cP = c; }
    
    else if(c.v[3] > P[3]) { subP(c.v, c.v); *cP = c; } 
    else if(c.v[3] < P[3]) { *cP = c; }
    
    else if(c.v[4] > P[4]) { subP(c.v, c.v); *cP = c; } 
    else if(c.v[4] < P[4]) { *cP = c; }
    
    else if(c.v[5] > P[5]) { subP(c.v, c.v); *cP = c; } 
    else if(c.v[5] < P[5]) { *cP = c; }
    
    else if(c.v[6] > P[6]) { subP(c.v, c.v); *cP = c; } 
    else if(c.v[6] < P[6]) { *cP = c; }

    else if(c.v[7] > P[7]) { subP(c.v, c.v); *cP = c; } 
    else { *cP = c; }
}


void mulModP(unsigned int a[8], unsigned int b[8], unsigned int product_low[8])
{
    unsigned int ZERO = 0;
    unsigned int high[8];

    unsigned int hWord = 0;
    unsigned int carry = 0;
    unsigned int t = 0;
    unsigned int product6 = 0;
    unsigned int product7 = 0;


    // 256 x 256 multiply
    multiply256(a, b, high, product_low);

    // Add 2^32 * high to the low 256 bits (shift left 1 word and add)
    // Affects product[14] to product[6]
    addc(&product_low[6], &high[7], &carry, &product_low[6]);
    addc(&product_low[5], &high[6], &carry, &product_low[5]);
    addc(&product_low[4], &high[5], &carry, &product_low[4]);
    addc(&product_low[3], &high[4], &carry, &product_low[3]);
    addc(&product_low[2], &high[3], &carry, &product_low[2]);
    addc(&product_low[1], &high[2], &carry, &product_low[1]);
    addc(&product_low[0], &high[1], &carry, &product_low[0]);

    addc(&high[0], &ZERO, &carry, &product7);
    product6 = carry;

    carry = 0;

    // Multiply high by 977 and add to low
    // Affects product[15] to product[5]
    for(int i = 7; i >= 0; i--) {
        madd977(&hWord, &t, &high[i], &hWord);
        addc(&product_low[i], &t, &carry, &product_low[i]);
        t = 0;
    }
    addc(&product7, &hWord, &carry, &product7);
    addc(&product6, &ZERO, &carry, &product6);

    // Multiply high 2 words by 2^32 and add to low
    // Affects product[14] to product[7]
    carry = 0;
    high[7] = product7;
    high[6] = product6;

    product7 = 0;
    product6 = 0;

    addc(&product_low[6], &high[7], &carry, &product_low[6]);
    addc(&product_low[5], &high[6], &carry, &product_low[5]);

    addc(&product_low[4], &ZERO, &carry, &product_low[4]);
    addc(&product_low[3], &ZERO, &carry, &product_low[3]);
    addc(&product_low[2], &ZERO, &carry, &product_low[2]);
    addc(&product_low[1], &ZERO, &carry, &product_low[1]);
    addc(&product_low[0], &ZERO, &carry, &product_low[0]);

    product7 = carry;

    // Multiply top 2 words by 977 and add to low
    // Affects product[15] to product[7]
    carry = 0;
    hWord = 0;
    madd977(&hWord, &t, &high[7], &hWord);
    addc(&product_low[7], &t, &carry, &product_low[7]);
    madd977(&hWord, &t, &high[6], &hWord);
    addc(&product_low[6], &t, &carry, &product_low[6]);
    addc(&product_low[5], &hWord, &carry, &product_low[5]);

    // Propagate carry
    addc(&product_low[4], &ZERO, &carry, &product_low[4]);
    addc(&product_low[3], &ZERO, &carry, &product_low[3]);
    addc(&product_low[2], &ZERO, &carry, &product_low[2]);
    addc(&product_low[1], &ZERO, &carry, &product_low[1]);
    addc(&product_low[0], &ZERO, &carry, &product_low[0]);
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

void mulModP256kv(uint256_t *a, uint256_t *b, uint256_t *c)
{
    mulModP(a->v, b->v, c->v);
}

void squareModP256k(uint256_t *a)
{
    mulModP(a->v, a->v, a->v);
}

/**
 * Multiplicative inverse mod P using Fermat's method of x^(p-2) mod p and addition chains
 */
uint256_t invModP256k(uint256_t x)
{
    uint256_t y = {{0, 0, 0, 0, 0, 0, 0, 1}};

    mulModP256kv(&x, &y, &y);
    squareModP256k(&x);
    squareModP256k(&x);
    mulModP256kv(&x, &y, &y);
    squareModP256k(&x);
    mulModP256kv(&x, &y, &y);
    squareModP256k(&x);
    squareModP256k(&x);
    mulModP256kv(&x, &y, &y);

    for(int i = 0; i < 5; i++) {
        squareModP256k(&x);
    }

    for(int i = 0; i < 22; i++) {
        mulModP256kv(&x, &y, &y);
        squareModP256k(&x);
    }

    squareModP256k(&x);

    for(int i = 0; i < 222; i++) {
        mulModP256kv(&x, &y, &y);
        squareModP256k(&x);
    }

    return mulModP256k(x, y);
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

    if(equal256k(&px, &x)) {
        addModP256k(&py,&py, &x);
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
        if(equal256k(&px, &x)) {
            addModP256k(&py, &py, &diff);
        } else {
            diff = subModP256k(px, x);
        }

        *inverse = mulModP256k(diff, *inverse);
    } else {
        s = *inverse;
    }


    if(equal256k(&px, &x)) {
        // currently s = 1 / 2y

        uint256_t x2;
        uint256_t tx2;

        // 3x^2
        mulModP256kv(&x, &x, &x2);
        addModP256k(&x2, &x2, &tx2);
        addModP256k(&x2, &tx2, &tx2);

        // s = 3x^2 * 1/2y
        mulModP256kv(&tx2, &s, &s);

        // s^2
        uint256_t s2;
        mulModP256kv(&s, &s, &s2);

        // Rx = s^2 - 2px
        *newX = subModP256k(s2, x);
        *newX = subModP256k(*newX, x);

        // Ry = s(px - rx) - py
        uint256_t k = subModP256k(px, *newX);
        *newY = mulModP256k(s, k);
        *newY = subModP256k(*newY, py);
    } else {

        uint256_t rise;
        rise = subModP256k(py, y);

        mulModP256kv(&rise, &s, &s);

        // Rx = s^2 - Gx - Qx
        uint256_t s2;
        mulModP256kv(&s, &s, &s2);

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

    uint256_t y = yPtr[i];

    uint256_t rise = subModP256k(py, y);

    s = mulModP256k(rise, s);

    // Rx = s^2 - Gx - Qx
    uint256_t s2;
    mulModP256kv(&s, &s, &s2);

    *newX = subModP256k(s2, px);
    *newX = subModP256k(*newX, x);

    // Ry = s(px - rx) - py
    uint256_t k = subModP256k(px, *newX);
    *newY = mulModP256k(s, k);
    *newY = subModP256k(*newY, py);
}


uint256_t doBatchInverse256k(uint256_t x)
{
    return invModP256k(x);
}

#endif
