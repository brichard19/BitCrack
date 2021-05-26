#ifndef RIPEMD160_CL
#define RIPEMD160_CL

#define RIPEMD160_IV_0 (0x67452301)
#define RIPEMD160_IV_1 (0xefcdab89)
#define RIPEMD160_IV_2 (0x98badcfe)
#define RIPEMD160_IV_3 (0x10325476)
#define RIPEMD160_IV_4 (0xc3d2e1f0)

#define rotl(x, n) (((x) << (n)) | ((x) >> (32 - (n))))

#define F(x, y, z) ((x) ^ (y) ^ (z))

#define G(x, y, z) (((x) & (y)) | (~(x) & (z)))

#define H(x, y, z) (((x) | ~(y)) ^ (z))

#define I(x, y, z) (((x) & (z)) | ((y) & ~(z)))

#define J(x, y, z) ((x) ^ ((y) | ~(z)))

#define FF(a, b, c, d, e, m, s)\
    a += (F((b), (c), (d)) + (m));\
    a = (rotl((a), (s)) + (e));\
    c = rotl((c), 10)

#define GG(a, b, c, d, e, x, s)\
    a += G((b), (c), (d)) + (x) + 0x5a827999;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define HH(a, b, c, d, e, x, s)\
    a += H((b), (c), (d)) + (x) + 0x6ed9eba1;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define II(a, b, c, d, e, x, s)\
    a += I((b), (c), (d)) + (x) + 0x8f1bbcdc;\
    a = rotl((a), (s)) + e;\
    c = rotl((c), 10)

#define JJ(a, b, c, d, e, x, s)\
    a += J((b), (c), (d)) + (x) + 0xa953fd4e;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define FFF(a, b, c, d, e, x, s)\
    a += F((b), (c), (d)) + (x);\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define GGG(a, b, c, d, e, x, s)\
    a += G((b), (c), (d)) + x + 0x7a6d76e9;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define HHH(a, b, c, d, e, x, s)\
    a += H((b), (c), (d)) + (x) + 0x6d703ef3;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define III(a, b, c, d, e, x, s)\
    a += I((b), (c), (d)) + (x) + 0x5c4dd124;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define JJJ(a, b, c, d, e, x, s)\
    a += J((b), (c), (d)) + (x) + 0x50a28be6;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

void ripemd160p1(const unsigned int x[8], unsigned int digest[5])
{
    unsigned int a = RIPEMD160_IV_0;
    unsigned int b = RIPEMD160_IV_1;
    unsigned int c = RIPEMD160_IV_2;
    unsigned int d = RIPEMD160_IV_3;
    unsigned int e = RIPEMD160_IV_4;

    /* round 1 */
    FF(a, b, c, d, e, x[0], 11);
    FF(e, a, b, c, d, x[1], 14);
    FF(d, e, a, b, c, x[2], 15);
    FF(c, d, e, a, b, x[3], 12);
    FF(b, c, d, e, a, x[4], 5);
    FF(a, b, c, d, e, x[5], 8);
    FF(e, a, b, c, d, x[6], 7);
    FF(d, e, a, b, c, x[7], 9);
    FF(c, d, e, a, b, 128, 11);
    FF(b, c, d, e, a, 0, 13);
    FF(a, b, c, d, e, 0, 14);
    FF(e, a, b, c, d, 0, 15);
    FF(d, e, a, b, c, 0, 6);
    FF(c, d, e, a, b, 0, 7);
    FF(b, c, d, e, a, 256, 9);
    FF(a, b, c, d, e, 0, 8);

    /* round 2 */
    GG(e, a, b, c, d, x[7], 7);
    GG(d, e, a, b, c, x[4], 6);
    GG(c, d, e, a, b, 0, 8);
    GG(b, c, d, e, a, x[1], 13);
    GG(a, b, c, d, e, 0, 11);
    GG(e, a, b, c, d, x[6], 9);
    GG(d, e, a, b, c, 0, 7);
    GG(c, d, e, a, b, x[3], 15);
    GG(b, c, d, e, a, 0, 7);
    GG(a, b, c, d, e, x[0], 12);
    GG(e, a, b, c, d, 0, 15);
    GG(d, e, a, b, c, x[5], 9);
    GG(c, d, e, a, b, x[2], 11);
    GG(b, c, d, e, a, 256, 7);
    GG(a, b, c, d, e, 0, 13);
    GG(e, a, b, c, d, 0x80, 12);

    /* round 3 */
    HH(d, e, a, b, c, x[3], 11);
    HH(c, d, e, a, b, 0, 13);
    HH(b, c, d, e, a, 256, 6);
    HH(a, b, c, d, e, x[4], 7);
    HH(e, a, b, c, d, 0, 14);
    HH(d, e, a, b, c, 0, 9);
    HH(c, d, e, a, b, 0x80, 13);
    HH(b, c, d, e, a, x[1], 15);
    HH(a, b, c, d, e, x[2], 14);
    HH(e, a, b, c, d, x[7], 8);
    HH(d, e, a, b, c, x[0], 13);
    HH(c, d, e, a, b, x[6], 6);
    HH(b, c, d, e, a, 0, 5);
    HH(a, b, c, d, e, 0, 12);
    HH(e, a, b, c, d, x[5], 7);
    HH(d, e, a, b, c, 0, 5);

    /* round 4 */
    II(c, d, e, a, b, x[1], 11);
    II(b, c, d, e, a, 0, 12);
    II(a, b, c, d, e, 0, 14);
    II(e, a, b, c, d, 0, 15);
    II(d, e, a, b, c, x[0], 14);
    II(c, d, e, a, b, 0x80, 15);
    II(b, c, d, e, a, 0, 9);
    II(a, b, c, d, e, x[4], 8);
    II(e, a, b, c, d, 0, 9);
    II(d, e, a, b, c, x[3], 14);
    II(c, d, e, a, b, x[7], 5);
    II(b, c, d, e, a, 0, 6);
    II(a, b, c, d, e, 256, 8);
    II(e, a, b, c, d, x[5], 6);
    II(d, e, a, b, c, x[6], 5);
    II(c, d, e, a, b, x[2], 12);

    /* round 5 */
    JJ(b, c, d, e, a, x[4], 9);
    JJ(a, b, c, d, e, x[0], 15);
    JJ(e, a, b, c, d, x[5], 5);
    JJ(d, e, a, b, c, 0, 11);
    JJ(c, d, e, a, b, x[7], 6);
    JJ(b, c, d, e, a, 0, 8);
    JJ(a, b, c, d, e, x[2], 13);
    JJ(e, a, b, c, d, 0, 12);
    JJ(d, e, a, b, c, 256, 5);
    JJ(c, d, e, a, b, x[1], 12);
    JJ(b, c, d, e, a, x[3], 13);
    JJ(a, b, c, d, e, 0x80, 14);
    JJ(e, a, b, c, d, 0, 11);
    JJ(d, e, a, b, c, x[6], 8);
    JJ(c, d, e, a, b, 0, 5);
    JJ(b, c, d, e, a, 0, 6);

    digest[0] = c;
    digest[1] = d;
    digest[2] = e;
    digest[3] = a;
    digest[4] = b;
}

void ripemd160p2(const unsigned int x[8], unsigned int digest[5])
{
    unsigned int a = RIPEMD160_IV_0;
    unsigned int b = RIPEMD160_IV_1;
    unsigned int c = RIPEMD160_IV_2;
    unsigned int d = RIPEMD160_IV_3;
    unsigned int e = RIPEMD160_IV_4;

    /* parallel round 1 */
    JJJ(a, b, c, d, e, x[5], 8);
    JJJ(e, a, b, c, d, 256, 9);
    JJJ(d, e, a, b, c, x[7], 9);
    JJJ(c, d, e, a, b, x[0], 11);
    JJJ(b, c, d, e, a, 0, 13);
    JJJ(a, b, c, d, e, x[2], 15);
    JJJ(e, a, b, c, d, 0, 15);
    JJJ(d, e, a, b, c, x[4], 5);
    JJJ(c, d, e, a, b, 0, 7);
    JJJ(b, c, d, e, a, x[6], 7);
    JJJ(a, b, c, d, e, 0, 8);
    JJJ(e, a, b, c, d, 0x80, 11);
    JJJ(d, e, a, b, c, x[1], 14);
    JJJ(c, d, e, a, b, 0, 14);
    JJJ(b, c, d, e, a, x[3], 12);
    JJJ(a, b, c, d, e, 0, 6);

    /* parallel round 2 */
    III(e, a, b, c, d, x[6], 9);
    III(d, e, a, b, c, 0, 13);
    III(c, d, e, a, b, x[3], 15);
    III(b, c, d, e, a, x[7], 7);
    III(a, b, c, d, e, x[0], 12);
    III(e, a, b, c, d, 0, 8);
    III(d, e, a, b, c, x[5], 9);
    III(c, d, e, a, b, 0, 11);
    III(b, c, d, e, a, 256, 7);
    III(a, b, c, d, e, 0, 7);
    III(e, a, b, c, d, 0x80, 12);
    III(d, e, a, b, c, 0, 7);
    III(c, d, e, a, b, x[4], 6);
    III(b, c, d, e, a, 0, 15);
    III(a, b, c, d, e, x[1], 13);
    III(e, a, b, c, d, x[2], 11);

    /* parallel round 3 */
    HHH(d, e, a, b, c, 0, 9);
    HHH(c, d, e, a, b, x[5], 7);
    HHH(b, c, d, e, a, x[1], 15);
    HHH(a, b, c, d, e, x[3], 11);
    HHH(e, a, b, c, d, x[7], 8);
    HHH(d, e, a, b, c, 256, 6);
    HHH(c, d, e, a, b, x[6], 6);
    HHH(b, c, d, e, a, 0, 14);
    HHH(a, b, c, d, e, 0, 12);
    HHH(e, a, b, c, d, 0x80, 13);
    HHH(d, e, a, b, c, 0, 5);
    HHH(c, d, e, a, b, x[2], 14);
    HHH(b, c, d, e, a, 0, 13);
    HHH(a, b, c, d, e, x[0], 13);
    HHH(e, a, b, c, d, x[4], 7);
    HHH(d, e, a, b, c, 0, 5);

    /* parallel round 4 */
    GGG(c, d, e, a, b, 0x80, 15);
    GGG(b, c, d, e, a, x[6], 5);
    GGG(a, b, c, d, e, x[4], 8);
    GGG(e, a, b, c, d, x[1], 11);
    GGG(d, e, a, b, c, x[3], 14);
    GGG(c, d, e, a, b, 0, 14);
    GGG(b, c, d, e, a, 0, 6);
    GGG(a, b, c, d, e, x[0], 14);
    GGG(e, a, b, c, d, x[5], 6);
    GGG(d, e, a, b, c, 0, 9);
    GGG(c, d, e, a, b, x[2], 12);
    GGG(b, c, d, e, a, 0, 9);
    GGG(a, b, c, d, e, 0, 12);
    GGG(e, a, b, c, d, x[7], 5);
    GGG(d, e, a, b, c, 0, 15);
    GGG(c, d, e, a, b, 256, 8);

    /* parallel round 5 */
    FFF(b, c, d, e, a, 0, 8);
    FFF(a, b, c, d, e, 0, 5);
    FFF(e, a, b, c, d, 0, 12);
    FFF(d, e, a, b, c, x[4], 9);
    FFF(c, d, e, a, b, x[1], 12);
    FFF(b, c, d, e, a, x[5], 5);
    FFF(a, b, c, d, e, 0x80, 14);
    FFF(e, a, b, c, d, x[7], 6);
    FFF(d, e, a, b, c, x[6], 8);
    FFF(c, d, e, a, b, x[2], 13);
    FFF(b, c, d, e, a, 0, 6);
    FFF(a, b, c, d, e, 256, 5);
    FFF(e, a, b, c, d, x[0], 15);
    FFF(d, e, a, b, c, x[3], 13);
    FFF(c, d, e, a, b, 0, 11);
    FFF(b, c, d, e, a, 0, 11);

    digest[0] = d;
    digest[1] = e;
    digest[2] = a;
    digest[3] = b;
    digest[4] = c;
}

void ripemd160sha256NoFinal(const unsigned int x[8], unsigned int digest[5])
{
    unsigned int digest1[5];
    unsigned int digest2[5];

    ripemd160p1(x, digest1);
    ripemd160p2(x, digest2);

    digest[0] = digest1[0] + digest2[0];
    digest[1] = digest1[1] + digest2[1];
    digest[2] = digest1[2] + digest2[2];
    digest[3] = digest1[3] + digest2[3];
    digest[4] = digest1[4] + digest2[4];

}

#endif
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

void mulModP256k(uint256_t *a, uint256_t *b, uint256_t *c)
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

    mulModP256k(&x, &y, &y);
    squareModP256k(&x);
    squareModP256k(&x);
    mulModP256k(&x, &y, &y);
    squareModP256k(&x);
    mulModP256k(&x, &y, &y);
    squareModP256k(&x);
    squareModP256k(&x);
    mulModP256k(&x, &y, &y);

    for(int i = 0; i < 5; i++) {
        squareModP256k(&x);
    }

    for(int i = 0; i < 22; i++) {
        mulModP256k(&x, &y, &y);
        squareModP256k(&x);
    }

    squareModP256k(&x);

    for(int i = 0; i < 222; i++) {
        mulModP256k(&x, &y, &y);
        squareModP256k(&x);
    }

    mulModP256k(&x, &y, &x);
    return x;
}


void beginBatchAdd256k(uint256_t px, uint256_t x, __global uint256_t* chain, int i, int batchIdx, uint256_t* inverse)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    // x = Gx - x
    uint256_t t = subModP256k(px, x);


    // Keep a chain of multiples of the diff, i.e. c[0] = diff0, c[1] = diff0 * diff1,
    // c[2] = diff2 * diff1 * diff0, etc
    mulModP256k(inverse, &t, inverse);

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
    mulModP256k(&x, inverse, inverse);

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
        mulModP256k(inverse, &c, &s);

        uint256_t diff;
        if(equal256k(&px, &x)) {
            addModP256k(&py, &py, &diff);
        } else {
            diff = subModP256k(px, x);
        }

        mulModP256k(&diff, inverse, inverse);
    } else {
        s = *inverse;
    }


    if(equal256k(&px, &x)) {
        // currently s = 1 / 2y

        uint256_t x2;
        uint256_t tx2;

        // 3x^2
        mulModP256k(&x, &x, &x2);
        addModP256k(&x2, &x2, &tx2);
        addModP256k(&x2, &tx2, &tx2);

        // s = 3x^2 * 1/2y
        mulModP256k(&tx2, &s, &s);

        // s^2
        uint256_t s2;
        mulModP256k(&s, &s, &s2);

        // Rx = s^2 - 2px
        *newX = subModP256k(s2, x);
        *newX = subModP256k(*newX, x);

        // Ry = s(px - rx) - py
        uint256_t k = subModP256k(px, *newX);
        mulModP256k(&s, &k, newY);
        *newY = subModP256k(*newY, py);
    } else {

        uint256_t rise;
        rise = subModP256k(py, y);

        mulModP256k(&rise, &s, &s);

        // Rx = s^2 - Gx - Qx
        uint256_t s2;
        mulModP256k(&s, &s, &s2);

        *newX = subModP256k(s2, px);
        *newX = subModP256k(*newX, x);

        // Ry = s(px - rx) - py
        uint256_t k;
        k = subModP256k(px, *newX);
        mulModP256k(&s, &k, newY);
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
        mulModP256k(inverse, &c, &s);

        uint256_t diff;
        diff = subModP256k(px, x);
        mulModP256k(&diff, inverse, inverse);
    } else {
        s = *inverse;
    }

    uint256_t y = yPtr[i];

    uint256_t rise = subModP256k(py, y);

    mulModP256k(&rise, &s, &s);

    // Rx = s^2 - Gx - Qx
    uint256_t s2;
    mulModP256k(&s, &s, &s2);

    *newX = subModP256k(s2, px);
    *newX = subModP256k(*newX, x);

    // Ry = s(px - rx) - py
    uint256_t k = subModP256k(px, *newX);
    mulModP256k(&s, &k, newY);
    *newY = subModP256k(*newY, py);
}


uint256_t doBatchInverse256k(uint256_t x)
{
    return invModP256k(x);
}

#endif
#ifndef _SHA256_CL
#define _SHA256_CL


__constant unsigned int _K[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant unsigned int _IV[8] = {
    0x6a09e667,
    0xbb67ae85,
    0x3c6ef372,
    0xa54ff53a,
    0x510e527f,
    0x9b05688c,
    0x1f83d9ab,
    0x5be0cd19
};

#define rotr(x, n) ((x) >> (n)) ^ ((x) << (32 - (n)))


#define MAJ(a, b, c) (((a) & (b)) ^ ((a) & (c)) ^ ((b) & (c)))

#define CH(e, f, g) (((e) & (f)) ^ (~(e) & (g)))

#define s0(x) (rotr((x), 7) ^ rotr((x), 18) ^ ((x) >> 3))

#define s1(x) (rotr((x), 17) ^ rotr((x), 19) ^ ((x) >> 10))

#define roundSha(a, b, c, d, e, f, g, h, m, k)\
    t = CH((e), (f), (g)) + (rotr((e), 6) ^ rotr((e), 11) ^ rotr((e), 25)) + (k) + (m);\
    (d) += (t) + (h);\
    (h) += (t) + MAJ((a), (b), (c)) + (rotr((a), 2) ^ rotr((a), 13) ^ rotr((a), 22))


void sha256PublicKey(const unsigned int x[8], const unsigned int y[8], unsigned int digest[8])
{
    unsigned int a, b, c, d, e, f, g, h;
    unsigned int w[16];
    unsigned int t;

    // 0x04 || x || y
    w[0] = (x[0] >> 8) | 0x04000000;
    w[1] = (x[1] >> 8) | (x[0] << 24);
    w[2] = (x[2] >> 8) | (x[1] << 24);
    w[3] = (x[3] >> 8) | (x[2] << 24);
    w[4] = (x[4] >> 8) | (x[3] << 24);
    w[5] = (x[5] >> 8) | (x[4] << 24);
    w[6] = (x[6] >> 8) | (x[5] << 24);
    w[7] = (x[7] >> 8) | (x[6] << 24);
    w[8] = (y[0] >> 8) | (x[7] << 24);
    w[9] = (y[1] >> 8) | (y[0] << 24);
    w[10] = (y[2] >> 8) | (y[1] << 24);
    w[11] = (y[3] >> 8) | (y[2] << 24);
    w[12] = (y[4] >> 8) | (y[3] << 24);
    w[13] = (y[5] >> 8) | (y[4] << 24);
    w[14] = (y[6] >> 8) | (y[5] << 24);
    w[15] = (y[7] >> 8) | (y[6] << 24);

    a = _IV[0];
    b = _IV[1];
    c = _IV[2];
    d = _IV[3];
    e = _IV[4];
    f = _IV[5];
    g = _IV[6];
    h = _IV[7];

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[0]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[1]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[2]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[3]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[4]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[5]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[6]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[7]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[8]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[9]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[10]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[11]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[12]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[13]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[14]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[15]);

    w[0] = w[0] + s0(w[1]) + w[9] + s1(w[14]);
    w[1] = w[1] + s0(w[2]) + w[10] + s1(w[15]);
    w[2] = w[2] + s0(w[3]) + w[11] + s1(w[0]);
    w[3] = w[3] + s0(w[4]) + w[12] + s1(w[1]);
    w[4] = w[4] + s0(w[5]) + w[13] + s1(w[2]);
    w[5] = w[5] + s0(w[6]) + w[14] + s1(w[3]);
    w[6] = w[6] + s0(w[7]) + w[15] + s1(w[4]);
    w[7] = w[7] + s0(w[8]) + w[0] + s1(w[5]);
    w[8] = w[8] + s0(w[9]) + w[1] + s1(w[6]);
    w[9] = w[9] + s0(w[10]) + w[2] + s1(w[7]);
    w[10] = w[10] + s0(w[11]) + w[3] + s1(w[8]);
    w[11] = w[11] + s0(w[12]) + w[4] + s1(w[9]);
    w[12] = w[12] + s0(w[13]) + w[5] + s1(w[10]);
    w[13] = w[13] + s0(w[14]) + w[6] + s1(w[11]);
    w[14] = w[14] + s0(w[15]) + w[7] + s1(w[12]);
    w[15] = w[15] + s0(w[0]) + w[8] + s1(w[13]);

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[16]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[17]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[18]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[19]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[20]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[21]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[22]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[23]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[24]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[25]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[26]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[27]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[28]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[29]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[30]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[31]);

    w[0] = w[0] + s0(w[1]) + w[9] + s1(w[14]);
    w[1] = w[1] + s0(w[2]) + w[10] + s1(w[15]);
    w[2] = w[2] + s0(w[3]) + w[11] + s1(w[0]);
    w[3] = w[3] + s0(w[4]) + w[12] + s1(w[1]);
    w[4] = w[4] + s0(w[5]) + w[13] + s1(w[2]);
    w[5] = w[5] + s0(w[6]) + w[14] + s1(w[3]);
    w[6] = w[6] + s0(w[7]) + w[15] + s1(w[4]);
    w[7] = w[7] + s0(w[8]) + w[0] + s1(w[5]);
    w[8] = w[8] + s0(w[9]) + w[1] + s1(w[6]);
    w[9] = w[9] + s0(w[10]) + w[2] + s1(w[7]);
    w[10] = w[10] + s0(w[11]) + w[3] + s1(w[8]);
    w[11] = w[11] + s0(w[12]) + w[4] + s1(w[9]);
    w[12] = w[12] + s0(w[13]) + w[5] + s1(w[10]);
    w[13] = w[13] + s0(w[14]) + w[6] + s1(w[11]);
    w[14] = w[14] + s0(w[15]) + w[7] + s1(w[12]);
    w[15] = w[15] + s0(w[0]) + w[8] + s1(w[13]);

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[32]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[33]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[34]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[35]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[36]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[37]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[38]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[39]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[40]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[41]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[42]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[43]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[44]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[45]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[46]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[47]);

    w[0] = w[0] + s0(w[1]) + w[9] + s1(w[14]);
    w[1] = w[1] + s0(w[2]) + w[10] + s1(w[15]);
    w[2] = w[2] + s0(w[3]) + w[11] + s1(w[0]);
    w[3] = w[3] + s0(w[4]) + w[12] + s1(w[1]);
    w[4] = w[4] + s0(w[5]) + w[13] + s1(w[2]);
    w[5] = w[5] + s0(w[6]) + w[14] + s1(w[3]);
    w[6] = w[6] + s0(w[7]) + w[15] + s1(w[4]);
    w[7] = w[7] + s0(w[8]) + w[0] + s1(w[5]);
    w[8] = w[8] + s0(w[9]) + w[1] + s1(w[6]);
    w[9] = w[9] + s0(w[10]) + w[2] + s1(w[7]);
    w[10] = w[10] + s0(w[11]) + w[3] + s1(w[8]);
    w[11] = w[11] + s0(w[12]) + w[4] + s1(w[9]);
    w[12] = w[12] + s0(w[13]) + w[5] + s1(w[10]);
    w[13] = w[13] + s0(w[14]) + w[6] + s1(w[11]);
    w[14] = w[14] + s0(w[15]) + w[7] + s1(w[12]);
    w[15] = w[15] + s0(w[0]) + w[8] + s1(w[13]);

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[48]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[49]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[50]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[51]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[52]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[53]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[54]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[55]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[56]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[57]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[58]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[59]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[60]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[61]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[62]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[63]);

    a += _IV[0];
    b += _IV[1];
    c += _IV[2];
    d += _IV[3];
    e += _IV[4];
    f += _IV[5];
    g += _IV[6];
    h += _IV[7];

    // store the intermediate hash value
    unsigned int tmp[8];
    tmp[0] = a;
    tmp[1] = b;
    tmp[2] = c;
    tmp[3] = d;
    tmp[4] = e;
    tmp[5] = f;
    tmp[6] = g;
    tmp[7] = h;

    w[0] = (y[7] << 24) | 0x00800000;
    w[15] = 65 * 8;

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[0]);
    roundSha(h, a, b, c, d, e, f, g, 0, _K[1]);
    roundSha(g, h, a, b, c, d, e, f, 0, _K[2]);
    roundSha(f, g, h, a, b, c, d, e, 0, _K[3]);
    roundSha(e, f, g, h, a, b, c, d, 0, _K[4]);
    roundSha(d, e, f, g, h, a, b, c, 0, _K[5]);
    roundSha(c, d, e, f, g, h, a, b, 0, _K[6]);
    roundSha(b, c, d, e, f, g, h, a, 0, _K[7]);
    roundSha(a, b, c, d, e, f, g, h, 0, _K[8]);
    roundSha(h, a, b, c, d, e, f, g, 0, _K[9]);
    roundSha(g, h, a, b, c, d, e, f, 0, _K[10]);
    roundSha(f, g, h, a, b, c, d, e, 0, _K[11]);
    roundSha(e, f, g, h, a, b, c, d, 0, _K[12]);
    roundSha(d, e, f, g, h, a, b, c, 0, _K[13]);
    roundSha(c, d, e, f, g, h, a, b, 0, _K[14]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[15]);

    w[0] = w[0] + s0(0) + 0 + s1(0);
    w[1] = 0 + s0(0) + 0 + s1(w[15]);
    w[2] = 0 + s0(0) + 0 + s1(w[0]);
    w[3] = 0 + s0(0) + 0 + s1(w[1]);
    w[4] = 0 + s0(0) + 0 + s1(w[2]);
    w[5] = 0 + s0(0) + 0 + s1(w[3]);
    w[6] = 0 + s0(0) + w[15] + s1(w[4]);
    w[7] = 0 + s0(0) + w[0] + s1(w[5]);
    w[8] = 0 + s0(0) + w[1] + s1(w[6]);
    w[9] = 0 + s0(0) + w[2] + s1(w[7]);
    w[10] = 0 + s0(0) + w[3] + s1(w[8]);
    w[11] = 0 + s0(0) + w[4] + s1(w[9]);
    w[12] = 0 + s0(0) + w[5] + s1(w[10]);
    w[13] = 0 + s0(0) + w[6] + s1(w[11]);
    w[14] = 0 + s0(w[15]) + w[7] + s1(w[12]);
    w[15] = w[15] + s0(w[0]) + w[8] + s1(w[13]);

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[16]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[17]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[18]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[19]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[20]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[21]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[22]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[23]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[24]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[25]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[26]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[27]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[28]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[29]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[30]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[31]);

    w[0] = w[0] + s0(w[1]) + w[9] + s1(w[14]);
    w[1] = w[1] + s0(w[2]) + w[10] + s1(w[15]);
    w[2] = w[2] + s0(w[3]) + w[11] + s1(w[0]);
    w[3] = w[3] + s0(w[4]) + w[12] + s1(w[1]);
    w[4] = w[4] + s0(w[5]) + w[13] + s1(w[2]);
    w[5] = w[5] + s0(w[6]) + w[14] + s1(w[3]);
    w[6] = w[6] + s0(w[7]) + w[15] + s1(w[4]);
    w[7] = w[7] + s0(w[8]) + w[0] + s1(w[5]);
    w[8] = w[8] + s0(w[9]) + w[1] + s1(w[6]);
    w[9] = w[9] + s0(w[10]) + w[2] + s1(w[7]);
    w[10] = w[10] + s0(w[11]) + w[3] + s1(w[8]);
    w[11] = w[11] + s0(w[12]) + w[4] + s1(w[9]);
    w[12] = w[12] + s0(w[13]) + w[5] + s1(w[10]);
    w[13] = w[13] + s0(w[14]) + w[6] + s1(w[11]);
    w[14] = w[14] + s0(w[15]) + w[7] + s1(w[12]);
    w[15] = w[15] + s0(w[0]) + w[8] + s1(w[13]);

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[32]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[33]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[34]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[35]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[36]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[37]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[38]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[39]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[40]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[41]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[42]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[43]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[44]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[45]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[46]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[47]);

    w[0] = w[0] + s0(w[1]) + w[9] + s1(w[14]);
    w[1] = w[1] + s0(w[2]) + w[10] + s1(w[15]);
    w[2] = w[2] + s0(w[3]) + w[11] + s1(w[0]);
    w[3] = w[3] + s0(w[4]) + w[12] + s1(w[1]);
    w[4] = w[4] + s0(w[5]) + w[13] + s1(w[2]);
    w[5] = w[5] + s0(w[6]) + w[14] + s1(w[3]);
    w[6] = w[6] + s0(w[7]) + w[15] + s1(w[4]);
    w[7] = w[7] + s0(w[8]) + w[0] + s1(w[5]);
    w[8] = w[8] + s0(w[9]) + w[1] + s1(w[6]);
    w[9] = w[9] + s0(w[10]) + w[2] + s1(w[7]);
    w[10] = w[10] + s0(w[11]) + w[3] + s1(w[8]);
    w[11] = w[11] + s0(w[12]) + w[4] + s1(w[9]);
    w[12] = w[12] + s0(w[13]) + w[5] + s1(w[10]);
    w[13] = w[13] + s0(w[14]) + w[6] + s1(w[11]);
    w[14] = w[14] + s0(w[15]) + w[7] + s1(w[12]);
    w[15] = w[15] + s0(w[0]) + w[8] + s1(w[13]);

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[48]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[49]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[50]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[51]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[52]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[53]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[54]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[55]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[56]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[57]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[58]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[59]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[60]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[61]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[62]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[63]);

    digest[0] = tmp[0] + a;
    digest[1] = tmp[1] + b;
    digest[2] = tmp[2] + c;
    digest[3] = tmp[3] + d;
    digest[4] = tmp[4] + e;
    digest[5] = tmp[5] + f;
    digest[6] = tmp[6] + g;
    digest[7] = tmp[7] + h;
}

void sha256PublicKeyCompressed(const unsigned int x[8], unsigned int yParity, unsigned int digest[8])
{
    unsigned int a, b, c, d, e, f, g, h;
    unsigned int w[16];
    unsigned int t;

    // 0x03 || x  or  0x02 || x
    w[0] = 0x02000000 | ((yParity & 1) << 24) | (x[0] >> 8);

    w[1] = (x[1] >> 8) | (x[0] << 24);
    w[2] = (x[2] >> 8) | (x[1] << 24);
    w[3] = (x[3] >> 8) | (x[2] << 24);
    w[4] = (x[4] >> 8) | (x[3] << 24);
    w[5] = (x[5] >> 8) | (x[4] << 24);
    w[6] = (x[6] >> 8) | (x[5] << 24);
    w[7] = (x[7] >> 8) | (x[6] << 24);
    w[8] = (x[7] << 24) | 0x00800000;
    w[15] = 33 * 8;

    a = _IV[0];
    b = _IV[1];
    c = _IV[2];
    d = _IV[3];
    e = _IV[4];
    f = _IV[5];
    g = _IV[6];
    h = _IV[7];

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[0]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[1]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[2]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[3]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[4]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[5]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[6]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[7]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[8]);
    roundSha(h, a, b, c, d, e, f, g, 0, _K[9]);
    roundSha(g, h, a, b, c, d, e, f, 0, _K[10]);
    roundSha(f, g, h, a, b, c, d, e, 0, _K[11]);
    roundSha(e, f, g, h, a, b, c, d, 0, _K[12]);
    roundSha(d, e, f, g, h, a, b, c, 0, _K[13]);
    roundSha(c, d, e, f, g, h, a, b, 0, _K[14]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[15]);

    w[0] = w[0] + s0(w[1]) + 0 + s1(0);
    w[1] = w[1] + s0(w[2]) + 0 + s1(w[15]);
    w[2] = w[2] + s0(w[3]) + 0 + s1(w[0]);
    w[3] = w[3] + s0(w[4]) + 0 + s1(w[1]);
    w[4] = w[4] + s0(w[5]) + 0 + s1(w[2]);
    w[5] = w[5] + s0(w[6]) + 0 + s1(w[3]);
    w[6] = w[6] + s0(w[7]) + w[15] + s1(w[4]);
    w[7] = w[7] + s0(w[8]) + w[0] + s1(w[5]);
    w[8] = w[8] + s0(0) + w[1] + s1(w[6]);
    w[9] = 0 + s0(0) + w[2] + s1(w[7]);
    w[10] = 0 + s0(0) + w[3] + s1(w[8]);
    w[11] = 0 + s0(0) + w[4] + s1(w[9]);
    w[12] = 0 + s0(0) + w[5] + s1(w[10]);
    w[13] = 0 + s0(0) + w[6] + s1(w[11]);
    w[14] = 0 + s0(w[15]) + w[7] + s1(w[12]);
    w[15] = w[15] + s0(w[0]) + w[8] + s1(w[13]);

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[16]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[17]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[18]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[19]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[20]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[21]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[22]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[23]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[24]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[25]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[26]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[27]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[28]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[29]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[30]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[31]);

    w[0] = w[0] + s0(w[1]) + w[9] + s1(w[14]);
    w[1] = w[1] + s0(w[2]) + w[10] + s1(w[15]);
    w[2] = w[2] + s0(w[3]) + w[11] + s1(w[0]);
    w[3] = w[3] + s0(w[4]) + w[12] + s1(w[1]);
    w[4] = w[4] + s0(w[5]) + w[13] + s1(w[2]);
    w[5] = w[5] + s0(w[6]) + w[14] + s1(w[3]);
    w[6] = w[6] + s0(w[7]) + w[15] + s1(w[4]);
    w[7] = w[7] + s0(w[8]) + w[0] + s1(w[5]);
    w[8] = w[8] + s0(w[9]) + w[1] + s1(w[6]);
    w[9] = w[9] + s0(w[10]) + w[2] + s1(w[7]);
    w[10] = w[10] + s0(w[11]) + w[3] + s1(w[8]);
    w[11] = w[11] + s0(w[12]) + w[4] + s1(w[9]);
    w[12] = w[12] + s0(w[13]) + w[5] + s1(w[10]);
    w[13] = w[13] + s0(w[14]) + w[6] + s1(w[11]);
    w[14] = w[14] + s0(w[15]) + w[7] + s1(w[12]);
    w[15] = w[15] + s0(w[0]) + w[8] + s1(w[13]);

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[32]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[33]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[34]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[35]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[36]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[37]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[38]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[39]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[40]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[41]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[42]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[43]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[44]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[45]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[46]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[47]);


    w[0] = w[0] + s0(w[1]) + w[9] + s1(w[14]);
    w[1] = w[1] + s0(w[2]) + w[10] + s1(w[15]);
    w[2] = w[2] + s0(w[3]) + w[11] + s1(w[0]);
    w[3] = w[3] + s0(w[4]) + w[12] + s1(w[1]);
    w[4] = w[4] + s0(w[5]) + w[13] + s1(w[2]);
    w[5] = w[5] + s0(w[6]) + w[14] + s1(w[3]);
    w[6] = w[6] + s0(w[7]) + w[15] + s1(w[4]);
    w[7] = w[7] + s0(w[8]) + w[0] + s1(w[5]);
    w[8] = w[8] + s0(w[9]) + w[1] + s1(w[6]);
    w[9] = w[9] + s0(w[10]) + w[2] + s1(w[7]);
    w[10] = w[10] + s0(w[11]) + w[3] + s1(w[8]);
    w[11] = w[11] + s0(w[12]) + w[4] + s1(w[9]);
    w[12] = w[12] + s0(w[13]) + w[5] + s1(w[10]);
    w[13] = w[13] + s0(w[14]) + w[6] + s1(w[11]);
    w[14] = w[14] + s0(w[15]) + w[7] + s1(w[12]);
    w[15] = w[15] + s0(w[0]) + w[8] + s1(w[13]);

    roundSha(a, b, c, d, e, f, g, h, w[0], _K[48]);
    roundSha(h, a, b, c, d, e, f, g, w[1], _K[49]);
    roundSha(g, h, a, b, c, d, e, f, w[2], _K[50]);
    roundSha(f, g, h, a, b, c, d, e, w[3], _K[51]);
    roundSha(e, f, g, h, a, b, c, d, w[4], _K[52]);
    roundSha(d, e, f, g, h, a, b, c, w[5], _K[53]);
    roundSha(c, d, e, f, g, h, a, b, w[6], _K[54]);
    roundSha(b, c, d, e, f, g, h, a, w[7], _K[55]);
    roundSha(a, b, c, d, e, f, g, h, w[8], _K[56]);
    roundSha(h, a, b, c, d, e, f, g, w[9], _K[57]);
    roundSha(g, h, a, b, c, d, e, f, w[10], _K[58]);
    roundSha(f, g, h, a, b, c, d, e, w[11], _K[59]);
    roundSha(e, f, g, h, a, b, c, d, w[12], _K[60]);
    roundSha(d, e, f, g, h, a, b, c, w[13], _K[61]);
    roundSha(c, d, e, f, g, h, a, b, w[14], _K[62]);
    roundSha(b, c, d, e, f, g, h, a, w[15], _K[63]);

    a += _IV[0];
    b += _IV[1];
    c += _IV[2];
    d += _IV[3];
    e += _IV[4];
    f += _IV[5];
    g += _IV[6];
    h += _IV[7];

    digest[0] = a;
    digest[1] = b;
    digest[2] = c;
    digest[3] = d;
    digest[4] = e;
    digest[5] = f;
    digest[6] = g;
    digest[7] = h;
}
#endif
#define COMPRESSED 0
#define UNCOMPRESSED 1
#define BOTH 2

unsigned int endian(unsigned int x)
{
    return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

typedef struct {
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
}CLDeviceResult;

bool isInBloomFilter(unsigned int hash[5], __global unsigned int *targetList, ulong mask)
{
    bool notFoundMatch = true;

    unsigned int h5 = hash[0] + hash[1] + hash[2] + hash[3] + hash[4];

    uint64_t idx[5];

    idx[0] = ((hash[0] << 6) | (h5 & 0x3f)) & mask;
    idx[1] = ((hash[1] << 6) | ((h5 >> 6) & 0x3f)) & mask;
    idx[2] = ((hash[2] << 6) | ((h5 >> 12) & 0x3f)) & mask;
    idx[3] = ((hash[3] << 6) | ((h5 >> 18) & 0x3f)) & mask;
    idx[4] = ((hash[4] << 6) | ((h5 >> 24) & 0x3f)) & mask;

    notFoundMatch = (targetList[idx[0] / 32] & (0x01 << (idx[0] % 32))) == 0
      || (targetList[idx[1] / 32] & (0x01 << (idx[1] % 32))) == 0
      || (targetList[idx[2] / 32] & (0x01 << (idx[2] % 32))) == 0
      || (targetList[idx[3] / 32] & (0x01 << (idx[3] % 32))) == 0
      || (targetList[idx[4] / 32] & (0x01 << (idx[4] % 32))) == 0;

    return notFoundMatch == false;
}

void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    hOut[0] = endian(hIn[0] + 0xefcdab89);
    hOut[1] = endian(hIn[1] + 0x98badcfe);
    hOut[2] = endian(hIn[2] + 0x10325476);
    hOut[3] = endian(hIn[3] + 0xc3d2e1f0);
    hOut[4] = endian(hIn[4] + 0x67452301);
}


__kernel void multiplyStepKernel(
    int totalPoints,
    int step,
    __global uint256_t* privateKeys,
    __global uint256_t* chain,
    __global uint256_t* gxPtr,
    __global uint256_t* gyPtr,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr)
{
    uint256_t gx;
    uint256_t gy;
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    gx = gxPtr[step];
    gy = gyPtr[step];

    // Multiply together all (_Gx - x) and then invert
    uint256_t inverse = { {0,0,0,0,0,0,0,1} };

    int batchIdx = 0;
    int i = gid;
    for(; i < totalPoints; i += dim) {

        unsigned int p;
        p = readWord256k(privateKeys, i, 7 - step / 32);

        unsigned int bit = p & (1 << (step % 32));

        uint256_t x = xPtr[i];

        if(bit != 0) {
            if(!isInfinity256k(&x)) {
                beginBatchAddWithDouble256k(gx, gy, xPtr, chain, i, batchIdx, &inverse);
                batchIdx++;
            }
        }
    }

    inverse = doBatchInverse256k(inverse);

    i -= dim;
    for(; i >= 0; i -= dim) {
        uint256_t newX;
        uint256_t newY;

        unsigned int p;
        p = readWord256k(privateKeys, i, 7 - step / 32);
        unsigned int bit = p & (1 << (step % 32));

        uint256_t x = xPtr[i];

        if(bit != 0) {
            if(!isInfinity256k(&x)) {
                batchIdx--;
                completeBatchAddWithDouble256k(gx, gy, xPtr, yPtr, i, batchIdx, chain, &inverse, &newX, &newY);
            } else {
                newX = gx;
                newY = gy;
            }

            xPtr[i] = newX;
            yPtr[i] = newY;
        }
    }
}


void hashPublicKey(uint256_t x, uint256_t y, unsigned int* digestOut)
{
    unsigned int hash[8];

    sha256PublicKey(x.v, y.v, hash);

    // Swap to little-endian
    hash[0] = endian(hash[0]);
    hash[1] = endian(hash[1]);
    hash[2] = endian(hash[2]);
    hash[3] = endian(hash[3]);
    hash[4] = endian(hash[4]);
    hash[5] = endian(hash[5]);
    hash[6] = endian(hash[6]);
    hash[7] = endian(hash[7]);

    ripemd160sha256NoFinal(hash, digestOut);
}

void hashPublicKeyCompressed(uint256_t x, unsigned int yParity, unsigned int* digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x.v, yParity, hash);

    // Swap to little-endian
    hash[0] = endian(hash[0]);
    hash[1] = endian(hash[1]);
    hash[2] = endian(hash[2]);
    hash[3] = endian(hash[3]);
    hash[4] = endian(hash[4]);
    hash[5] = endian(hash[5]);
    hash[6] = endian(hash[6]);
    hash[7] = endian(hash[7]);

    ripemd160sha256NoFinal(hash, digestOut);

}

void atomicListAdd(__global CLDeviceResult *results, __global unsigned int *numResults, CLDeviceResult *r)
{
    unsigned int count = atomic_add(numResults, 1);

    results[count] = *r;
}

void setResultFound(int idx, bool compressed, uint256_t x, uint256_t y, unsigned int digest[5], __global CLDeviceResult* results, __global unsigned int* numResults)
{
    CLDeviceResult r;

    r.idx = idx;
    r.compressed = compressed;

    r.x[0] = x.v[0];
    r.y[0] = y.v[0];

    r.x[1] = x.v[1];
    r.y[1] = y.v[1];
    
    r.x[2] = x.v[2];
    r.y[2] = y.v[2];
    
    r.x[3] = x.v[3];
    r.y[3] = y.v[3];

    r.x[4] = x.v[4];
    r.y[4] = y.v[4];
    
    r.x[5] = x.v[5];
    r.y[5] = y.v[5];
    
    r.x[6] = x.v[6];
    r.y[6] = y.v[6];
    
    r.x[7] = x.v[7];
    r.y[7] = y.v[7];

    doRMD160FinalRound(digest, r.digest);

    atomicListAdd(results, numResults, &r);
}

__kernel void keyFinderKernel(
    unsigned int totalPoints,
    int compression,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int* targetList,
    ulong numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    uint256_t incX = *incXPtr;
    uint256_t incY = *incYPtr;

    // Multiply together all (_Gx - x) and then invert
    uint256_t inverse = { {0,0,0,0,0,0,0,1} };
    int i = gid;
    int batchIdx = 0;

    for(; i < totalPoints; i += dim) {
        uint256_t x;

        unsigned int digest[5];

        x = xPtr[i];

        if((compression == UNCOMPRESSED) || (compression == BOTH)) {
            uint256_t y = yPtr[i];

            hashPublicKey(x, y, digest);

            if(isInBloomFilter(digest, targetList, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW256k(yPtr, i), digest);

            if(isInBloomFilter(digest, targetList, mask)) {
                uint256_t y = yPtr[i];
                setResultFound(i, true, x, y, digest, results, numResults);
            }
        }

        beginBatchAdd256k(incX, x, chain, i, batchIdx, &inverse);
        batchIdx++;
    }

    inverse = doBatchInverse256k(inverse);

    i -= dim;

    for(;  i >= 0; i -= dim) {

        uint256_t newX;
        uint256_t newY;
        batchIdx--;
        completeBatchAdd256k(incX, incY, xPtr, yPtr, i, batchIdx, chain, &inverse, &newX, &newY);

        xPtr[i] = newX;
        yPtr[i] = newY;
    }
}

__kernel void keyFinderKernelWithDouble(
    unsigned int totalPoints,
    int compression,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int* targetList,
    ulong numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    int gid = get_local_size(0) * get_group_id(0) + get_local_id(0);
    int dim = get_global_size(0);

    uint256_t incX = *incXPtr;
    uint256_t incY = *incYPtr;

    // Multiply together all (_Gx - x) and then invert
    uint256_t inverse = { {0,0,0,0,0,0,0,1} };

    int i = gid;
    int batchIdx = 0;
    for(; i < totalPoints; i += dim) {
        uint256_t x;

        unsigned int digest[5];

        x = xPtr[i];

        // uncompressed
        if((compression == UNCOMPRESSED) || (compression == BOTH)) {
            uint256_t y = yPtr[i];
            hashPublicKey(x, y, digest);

            if(isInBloomFilter(digest, targetList, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        // compressed
        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW256k(yPtr, i), digest);

            if(isInBloomFilter(digest, targetList, mask)) {

                uint256_t y = yPtr[i];
                setResultFound(i, true, x, y, digest, results, numResults);
            }
        }

        beginBatchAddWithDouble256k(incX, incY, xPtr, chain, i, batchIdx, &inverse);
        batchIdx++;
    }

    inverse = doBatchInverse256k(inverse);

    i -= dim;

    for(; i >= 0; i -= dim) {
        uint256_t newX;
        uint256_t newY;
        batchIdx--;
        completeBatchAddWithDouble256k(incX, incY, xPtr, yPtr, i, batchIdx, chain, &inverse, &newX, &newY);

        xPtr[i] = newX;
        yPtr[i] = newY;
    }
}
