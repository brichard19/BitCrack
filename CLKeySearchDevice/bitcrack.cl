#ifndef _RIPEMD160_CL
#define _RIPEMD160_CL


__constant unsigned int _RIPEMD160_IV[5] = {
    0x67452301,
    0xefcdab89,
    0x98badcfe,
    0x10325476,
    0xc3d2e1f0
};

__constant unsigned int _K0 = 0x5a827999;
__constant unsigned int _K1 = 0x6ed9eba1;
__constant unsigned int _K2 = 0x8f1bbcdc;
__constant unsigned int _K3 = 0xa953fd4e;

__constant unsigned int _K4 = 0x7a6d76e9;
__constant unsigned int _K5 = 0x6d703ef3;
__constant unsigned int _K6 = 0x5c4dd124;
__constant unsigned int _K7 = 0x50a28be6;

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
    a += G((b), (c), (d)) + (x) + _K0;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define HH(a, b, c, d, e, x, s)\
    a += H((b), (c), (d)) + (x) + _K1;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define II(a, b, c, d, e, x, s)\
    a += I((b), (c), (d)) + (x) + _K2;\
    a = rotl((a), (s)) + e;\
    c = rotl((c), 10)

#define JJ(a, b, c, d, e, x, s)\
    a += J((b), (c), (d)) + (x) + _K3;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define FFF(a, b, c, d, e, x, s)\
    a += F((b), (c), (d)) + (x);\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define GGG(a, b, c, d, e, x, s)\
    a += G((b), (c), (d)) + x + _K4;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define HHH(a, b, c, d, e, x, s)\
    a += H((b), (c), (d)) + (x) + _K5;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define III(a, b, c, d, e, x, s)\
    a += I((b), (c), (d)) + (x) + _K6;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define JJJ(a, b, c, d, e, x, s)\
    a += J((b), (c), (d)) + (x) + _K7;\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)


void ripemd160sha256(const unsigned int x[8], unsigned int digest[5])
{
    unsigned int a1 = _RIPEMD160_IV[0];
    unsigned int b1 = _RIPEMD160_IV[1];
    unsigned int c1 = _RIPEMD160_IV[2];
    unsigned int d1 = _RIPEMD160_IV[3];
    unsigned int e1 = _RIPEMD160_IV[4];

    const unsigned int x8 = 0x00000080;
    const unsigned int x14 = 256;

    /* round 1 */
    FF(a1, b1, c1, d1, e1, x[0], 11);
    FF(e1, a1, b1, c1, d1, x[1], 14);
    FF(d1, e1, a1, b1, c1, x[2], 15);
    FF(c1, d1, e1, a1, b1, x[3], 12);
    FF(b1, c1, d1, e1, a1, x[4], 5);
    FF(a1, b1, c1, d1, e1, x[5], 8);
    FF(e1, a1, b1, c1, d1, x[6], 7);
    FF(d1, e1, a1, b1, c1, x[7], 9);
    FF(c1, d1, e1, a1, b1, x8, 11);
    FF(b1, c1, d1, e1, a1, 0, 13);
    FF(a1, b1, c1, d1, e1, 0, 14);
    FF(e1, a1, b1, c1, d1, 0, 15);
    FF(d1, e1, a1, b1, c1, 0, 6);
    FF(c1, d1, e1, a1, b1, 0, 7);
    FF(b1, c1, d1, e1, a1, x14, 9);
    FF(a1, b1, c1, d1, e1, 0, 8);

    /* round 2 */
    GG(e1, a1, b1, c1, d1, x[7], 7);
    GG(d1, e1, a1, b1, c1, x[4], 6);
    GG(c1, d1, e1, a1, b1, 0, 8);
    GG(b1, c1, d1, e1, a1, x[1], 13);
    GG(a1, b1, c1, d1, e1, 0, 11);
    GG(e1, a1, b1, c1, d1, x[6], 9);
    GG(d1, e1, a1, b1, c1, 0, 7);
    GG(c1, d1, e1, a1, b1, x[3], 15);
    GG(b1, c1, d1, e1, a1, 0, 7);
    GG(a1, b1, c1, d1, e1, x[0], 12);
    GG(e1, a1, b1, c1, d1, 0, 15);
    GG(d1, e1, a1, b1, c1, x[5], 9);
    GG(c1, d1, e1, a1, b1, x[2], 11);
    GG(b1, c1, d1, e1, a1, x14, 7);
    GG(a1, b1, c1, d1, e1, 0, 13);
    GG(e1, a1, b1, c1, d1, x8, 12);

    /* round 3 */
    HH(d1, e1, a1, b1, c1, x[3], 11);
    HH(c1, d1, e1, a1, b1, 0, 13);
    HH(b1, c1, d1, e1, a1, x14, 6);
    HH(a1, b1, c1, d1, e1, x[4], 7);
    HH(e1, a1, b1, c1, d1, 0, 14);
    HH(d1, e1, a1, b1, c1, 0, 9);
    HH(c1, d1, e1, a1, b1, x8, 13);
    HH(b1, c1, d1, e1, a1, x[1], 15);
    HH(a1, b1, c1, d1, e1, x[2], 14);
    HH(e1, a1, b1, c1, d1, x[7], 8);
    HH(d1, e1, a1, b1, c1, x[0], 13);
    HH(c1, d1, e1, a1, b1, x[6], 6);
    HH(b1, c1, d1, e1, a1, 0, 5);
    HH(a1, b1, c1, d1, e1, 0, 12);
    HH(e1, a1, b1, c1, d1, x[5], 7);
    HH(d1, e1, a1, b1, c1, 0, 5);

    /* round 4 */
    II(c1, d1, e1, a1, b1, x[1], 11);
    II(b1, c1, d1, e1, a1, 0, 12);
    II(a1, b1, c1, d1, e1, 0, 14);
    II(e1, a1, b1, c1, d1, 0, 15);
    II(d1, e1, a1, b1, c1, x[0], 14);
    II(c1, d1, e1, a1, b1, x8, 15);
    II(b1, c1, d1, e1, a1, 0, 9);
    II(a1, b1, c1, d1, e1, x[4], 8);
    II(e1, a1, b1, c1, d1, 0, 9);
    II(d1, e1, a1, b1, c1, x[3], 14);
    II(c1, d1, e1, a1, b1, x[7], 5);
    II(b1, c1, d1, e1, a1, 0, 6);
    II(a1, b1, c1, d1, e1, x14, 8);
    II(e1, a1, b1, c1, d1, x[5], 6);
    II(d1, e1, a1, b1, c1, x[6], 5);
    II(c1, d1, e1, a1, b1, x[2], 12);

    /* round 5 */
    JJ(b1, c1, d1, e1, a1, x[4], 9);
    JJ(a1, b1, c1, d1, e1, x[0], 15);
    JJ(e1, a1, b1, c1, d1, x[5], 5);
    JJ(d1, e1, a1, b1, c1, 0, 11);
    JJ(c1, d1, e1, a1, b1, x[7], 6);
    JJ(b1, c1, d1, e1, a1, 0, 8);
    JJ(a1, b1, c1, d1, e1, x[2], 13);
    JJ(e1, a1, b1, c1, d1, 0, 12);
    JJ(d1, e1, a1, b1, c1, x14, 5);
    JJ(c1, d1, e1, a1, b1, x[1], 12);
    JJ(b1, c1, d1, e1, a1, x[3], 13);
    JJ(a1, b1, c1, d1, e1, x8, 14);
    JJ(e1, a1, b1, c1, d1, 0, 11);
    JJ(d1, e1, a1, b1, c1, x[6], 8);
    JJ(c1, d1, e1, a1, b1, 0, 5);
    JJ(b1, c1, d1, e1, a1, 0, 6);

    unsigned int a2 = _RIPEMD160_IV[0];
    unsigned int b2 = _RIPEMD160_IV[1];
    unsigned int c2 = _RIPEMD160_IV[2];
    unsigned int d2 = _RIPEMD160_IV[3];
    unsigned int e2 = _RIPEMD160_IV[4];

    /* parallel round 1 */
    JJJ(a2, b2, c2, d2, e2, x[5], 8);
    JJJ(e2, a2, b2, c2, d2, x14, 9);
    JJJ(d2, e2, a2, b2, c2, x[7], 9);
    JJJ(c2, d2, e2, a2, b2, x[0], 11);
    JJJ(b2, c2, d2, e2, a2, 0, 13);
    JJJ(a2, b2, c2, d2, e2, x[2], 15);
    JJJ(e2, a2, b2, c2, d2, 0, 15);
    JJJ(d2, e2, a2, b2, c2, x[4], 5);
    JJJ(c2, d2, e2, a2, b2, 0, 7);
    JJJ(b2, c2, d2, e2, a2, x[6], 7);
    JJJ(a2, b2, c2, d2, e2, 0, 8);
    JJJ(e2, a2, b2, c2, d2, x8, 11);
    JJJ(d2, e2, a2, b2, c2, x[1], 14);
    JJJ(c2, d2, e2, a2, b2, 0, 14);
    JJJ(b2, c2, d2, e2, a2, x[3], 12);
    JJJ(a2, b2, c2, d2, e2, 0, 6);

    /* parallel round 2 */
    III(e2, a2, b2, c2, d2, x[6], 9);
    III(d2, e2, a2, b2, c2, 0, 13);
    III(c2, d2, e2, a2, b2, x[3], 15);
    III(b2, c2, d2, e2, a2, x[7], 7);
    III(a2, b2, c2, d2, e2, x[0], 12);
    III(e2, a2, b2, c2, d2, 0, 8);
    III(d2, e2, a2, b2, c2, x[5], 9);
    III(c2, d2, e2, a2, b2, 0, 11);
    III(b2, c2, d2, e2, a2, x14, 7);
    III(a2, b2, c2, d2, e2, 0, 7);
    III(e2, a2, b2, c2, d2, x8, 12);
    III(d2, e2, a2, b2, c2, 0, 7);
    III(c2, d2, e2, a2, b2, x[4], 6);
    III(b2, c2, d2, e2, a2, 0, 15);
    III(a2, b2, c2, d2, e2, x[1], 13);
    III(e2, a2, b2, c2, d2, x[2], 11);

    /* parallel round 3 */
    HHH(d2, e2, a2, b2, c2, 0, 9);
    HHH(c2, d2, e2, a2, b2, x[5], 7);
    HHH(b2, c2, d2, e2, a2, x[1], 15);
    HHH(a2, b2, c2, d2, e2, x[3], 11);
    HHH(e2, a2, b2, c2, d2, x[7], 8);
    HHH(d2, e2, a2, b2, c2, x14, 6);
    HHH(c2, d2, e2, a2, b2, x[6], 6);
    HHH(b2, c2, d2, e2, a2, 0, 14);
    HHH(a2, b2, c2, d2, e2, 0, 12);
    HHH(e2, a2, b2, c2, d2, x8, 13);
    HHH(d2, e2, a2, b2, c2, 0, 5);
    HHH(c2, d2, e2, a2, b2, x[2], 14);
    HHH(b2, c2, d2, e2, a2, 0, 13);
    HHH(a2, b2, c2, d2, e2, x[0], 13);
    HHH(e2, a2, b2, c2, d2, x[4], 7);
    HHH(d2, e2, a2, b2, c2, 0, 5);

    /* parallel round 4 */
    GGG(c2, d2, e2, a2, b2, x8, 15);
    GGG(b2, c2, d2, e2, a2, x[6], 5);
    GGG(a2, b2, c2, d2, e2, x[4], 8);
    GGG(e2, a2, b2, c2, d2, x[1], 11);
    GGG(d2, e2, a2, b2, c2, x[3], 14);
    GGG(c2, d2, e2, a2, b2, 0, 14);
    GGG(b2, c2, d2, e2, a2, 0, 6);
    GGG(a2, b2, c2, d2, e2, x[0], 14);
    GGG(e2, a2, b2, c2, d2, x[5], 6);
    GGG(d2, e2, a2, b2, c2, 0, 9);
    GGG(c2, d2, e2, a2, b2, x[2], 12);
    GGG(b2, c2, d2, e2, a2, 0, 9);
    GGG(a2, b2, c2, d2, e2, 0, 12);
    GGG(e2, a2, b2, c2, d2, x[7], 5);
    GGG(d2, e2, a2, b2, c2, 0, 15);
    GGG(c2, d2, e2, a2, b2, x14, 8);

    /* parallel round 5 */
    FFF(b2, c2, d2, e2, a2, 0, 8);
    FFF(a2, b2, c2, d2, e2, 0, 5);
    FFF(e2, a2, b2, c2, d2, 0, 12);
    FFF(d2, e2, a2, b2, c2, x[4], 9);
    FFF(c2, d2, e2, a2, b2, x[1], 12);
    FFF(b2, c2, d2, e2, a2, x[5], 5);
    FFF(a2, b2, c2, d2, e2, x8, 14);
    FFF(e2, a2, b2, c2, d2, x[7], 6);
    FFF(d2, e2, a2, b2, c2, x[6], 8);
    FFF(c2, d2, e2, a2, b2, x[2], 13);
    FFF(b2, c2, d2, e2, a2, 0, 6);
    FFF(a2, b2, c2, d2, e2, x14, 5);
    FFF(e2, a2, b2, c2, d2, x[0], 15);
    FFF(d2, e2, a2, b2, c2, x[3], 13);
    FFF(c2, d2, e2, a2, b2, 0, 11);
    FFF(b2, c2, d2, e2, a2, 0, 11);

    digest[0] = _RIPEMD160_IV[1] + c1 + d2;
    digest[1] = _RIPEMD160_IV[2] + d1 + e2;
    digest[2] = _RIPEMD160_IV[3] + e1 + a2;
    digest[3] = _RIPEMD160_IV[4] + a1 + b2;
    digest[4] = _RIPEMD160_IV[0] + b1 + c2;
}


void ripemd160sha256NoFinal(const unsigned int x[8], unsigned int digest[5])
{
    unsigned int a1 = _RIPEMD160_IV[0];
    unsigned int b1 = _RIPEMD160_IV[1];
    unsigned int c1 = _RIPEMD160_IV[2];
    unsigned int d1 = _RIPEMD160_IV[3];
    unsigned int e1 = _RIPEMD160_IV[4];

    const unsigned int x8 = 0x00000080;
    const unsigned int x14 = 256;

    /* round 1 */
    FF(a1, b1, c1, d1, e1, x[0], 11);
    FF(e1, a1, b1, c1, d1, x[1], 14);
    FF(d1, e1, a1, b1, c1, x[2], 15);
    FF(c1, d1, e1, a1, b1, x[3], 12);
    FF(b1, c1, d1, e1, a1, x[4], 5);
    FF(a1, b1, c1, d1, e1, x[5], 8);
    FF(e1, a1, b1, c1, d1, x[6], 7);
    FF(d1, e1, a1, b1, c1, x[7], 9);
    FF(c1, d1, e1, a1, b1, x8, 11);
    FF(b1, c1, d1, e1, a1, 0, 13);
    FF(a1, b1, c1, d1, e1, 0, 14);
    FF(e1, a1, b1, c1, d1, 0, 15);
    FF(d1, e1, a1, b1, c1, 0, 6);
    FF(c1, d1, e1, a1, b1, 0, 7);
    FF(b1, c1, d1, e1, a1, x14, 9);
    FF(a1, b1, c1, d1, e1, 0, 8);

    /* round 2 */
    GG(e1, a1, b1, c1, d1, x[7], 7);
    GG(d1, e1, a1, b1, c1, x[4], 6);
    GG(c1, d1, e1, a1, b1, 0, 8);
    GG(b1, c1, d1, e1, a1, x[1], 13);
    GG(a1, b1, c1, d1, e1, 0, 11);
    GG(e1, a1, b1, c1, d1, x[6], 9);
    GG(d1, e1, a1, b1, c1, 0, 7);
    GG(c1, d1, e1, a1, b1, x[3], 15);
    GG(b1, c1, d1, e1, a1, 0, 7);
    GG(a1, b1, c1, d1, e1, x[0], 12);
    GG(e1, a1, b1, c1, d1, 0, 15);
    GG(d1, e1, a1, b1, c1, x[5], 9);
    GG(c1, d1, e1, a1, b1, x[2], 11);
    GG(b1, c1, d1, e1, a1, x14, 7);
    GG(a1, b1, c1, d1, e1, 0, 13);
    GG(e1, a1, b1, c1, d1, x8, 12);

    /* round 3 */
    HH(d1, e1, a1, b1, c1, x[3], 11);
    HH(c1, d1, e1, a1, b1, 0, 13);
    HH(b1, c1, d1, e1, a1, x14, 6);
    HH(a1, b1, c1, d1, e1, x[4], 7);
    HH(e1, a1, b1, c1, d1, 0, 14);
    HH(d1, e1, a1, b1, c1, 0, 9);
    HH(c1, d1, e1, a1, b1, x8, 13);
    HH(b1, c1, d1, e1, a1, x[1], 15);
    HH(a1, b1, c1, d1, e1, x[2], 14);
    HH(e1, a1, b1, c1, d1, x[7], 8);
    HH(d1, e1, a1, b1, c1, x[0], 13);
    HH(c1, d1, e1, a1, b1, x[6], 6);
    HH(b1, c1, d1, e1, a1, 0, 5);
    HH(a1, b1, c1, d1, e1, 0, 12);
    HH(e1, a1, b1, c1, d1, x[5], 7);
    HH(d1, e1, a1, b1, c1, 0, 5);

    /* round 4 */
    II(c1, d1, e1, a1, b1, x[1], 11);
    II(b1, c1, d1, e1, a1, 0, 12);
    II(a1, b1, c1, d1, e1, 0, 14);
    II(e1, a1, b1, c1, d1, 0, 15);
    II(d1, e1, a1, b1, c1, x[0], 14);
    II(c1, d1, e1, a1, b1, x8, 15);
    II(b1, c1, d1, e1, a1, 0, 9);
    II(a1, b1, c1, d1, e1, x[4], 8);
    II(e1, a1, b1, c1, d1, 0, 9);
    II(d1, e1, a1, b1, c1, x[3], 14);
    II(c1, d1, e1, a1, b1, x[7], 5);
    II(b1, c1, d1, e1, a1, 0, 6);
    II(a1, b1, c1, d1, e1, x14, 8);
    II(e1, a1, b1, c1, d1, x[5], 6);
    II(d1, e1, a1, b1, c1, x[6], 5);
    II(c1, d1, e1, a1, b1, x[2], 12);

    /* round 5 */
    JJ(b1, c1, d1, e1, a1, x[4], 9);
    JJ(a1, b1, c1, d1, e1, x[0], 15);
    JJ(e1, a1, b1, c1, d1, x[5], 5);
    JJ(d1, e1, a1, b1, c1, 0, 11);
    JJ(c1, d1, e1, a1, b1, x[7], 6);
    JJ(b1, c1, d1, e1, a1, 0, 8);
    JJ(a1, b1, c1, d1, e1, x[2], 13);
    JJ(e1, a1, b1, c1, d1, 0, 12);
    JJ(d1, e1, a1, b1, c1, x14, 5);
    JJ(c1, d1, e1, a1, b1, x[1], 12);
    JJ(b1, c1, d1, e1, a1, x[3], 13);
    JJ(a1, b1, c1, d1, e1, x8, 14);
    JJ(e1, a1, b1, c1, d1, 0, 11);
    JJ(d1, e1, a1, b1, c1, x[6], 8);
    JJ(c1, d1, e1, a1, b1, 0, 5);
    JJ(b1, c1, d1, e1, a1, 0, 6);

    unsigned int a2 = _RIPEMD160_IV[0];
    unsigned int b2 = _RIPEMD160_IV[1];
    unsigned int c2 = _RIPEMD160_IV[2];
    unsigned int d2 = _RIPEMD160_IV[3];
    unsigned int e2 = _RIPEMD160_IV[4];

    /* parallel round 1 */
    JJJ(a2, b2, c2, d2, e2, x[5], 8);
    JJJ(e2, a2, b2, c2, d2, x14, 9);
    JJJ(d2, e2, a2, b2, c2, x[7], 9);
    JJJ(c2, d2, e2, a2, b2, x[0], 11);
    JJJ(b2, c2, d2, e2, a2, 0, 13);
    JJJ(a2, b2, c2, d2, e2, x[2], 15);
    JJJ(e2, a2, b2, c2, d2, 0, 15);
    JJJ(d2, e2, a2, b2, c2, x[4], 5);
    JJJ(c2, d2, e2, a2, b2, 0, 7);
    JJJ(b2, c2, d2, e2, a2, x[6], 7);
    JJJ(a2, b2, c2, d2, e2, 0, 8);
    JJJ(e2, a2, b2, c2, d2, x8, 11);
    JJJ(d2, e2, a2, b2, c2, x[1], 14);
    JJJ(c2, d2, e2, a2, b2, 0, 14);
    JJJ(b2, c2, d2, e2, a2, x[3], 12);
    JJJ(a2, b2, c2, d2, e2, 0, 6);

    /* parallel round 2 */
    III(e2, a2, b2, c2, d2, x[6], 9);
    III(d2, e2, a2, b2, c2, 0, 13);
    III(c2, d2, e2, a2, b2, x[3], 15);
    III(b2, c2, d2, e2, a2, x[7], 7);
    III(a2, b2, c2, d2, e2, x[0], 12);
    III(e2, a2, b2, c2, d2, 0, 8);
    III(d2, e2, a2, b2, c2, x[5], 9);
    III(c2, d2, e2, a2, b2, 0, 11);
    III(b2, c2, d2, e2, a2, x14, 7);
    III(a2, b2, c2, d2, e2, 0, 7);
    III(e2, a2, b2, c2, d2, x8, 12);
    III(d2, e2, a2, b2, c2, 0, 7);
    III(c2, d2, e2, a2, b2, x[4], 6);
    III(b2, c2, d2, e2, a2, 0, 15);
    III(a2, b2, c2, d2, e2, x[1], 13);
    III(e2, a2, b2, c2, d2, x[2], 11);

    /* parallel round 3 */
    HHH(d2, e2, a2, b2, c2, 0, 9);
    HHH(c2, d2, e2, a2, b2, x[5], 7);
    HHH(b2, c2, d2, e2, a2, x[1], 15);
    HHH(a2, b2, c2, d2, e2, x[3], 11);
    HHH(e2, a2, b2, c2, d2, x[7], 8);
    HHH(d2, e2, a2, b2, c2, x14, 6);
    HHH(c2, d2, e2, a2, b2, x[6], 6);
    HHH(b2, c2, d2, e2, a2, 0, 14);
    HHH(a2, b2, c2, d2, e2, 0, 12);
    HHH(e2, a2, b2, c2, d2, x8, 13);
    HHH(d2, e2, a2, b2, c2, 0, 5);
    HHH(c2, d2, e2, a2, b2, x[2], 14);
    HHH(b2, c2, d2, e2, a2, 0, 13);
    HHH(a2, b2, c2, d2, e2, x[0], 13);
    HHH(e2, a2, b2, c2, d2, x[4], 7);
    HHH(d2, e2, a2, b2, c2, 0, 5);

    /* parallel round 4 */
    GGG(c2, d2, e2, a2, b2, x8, 15);
    GGG(b2, c2, d2, e2, a2, x[6], 5);
    GGG(a2, b2, c2, d2, e2, x[4], 8);
    GGG(e2, a2, b2, c2, d2, x[1], 11);
    GGG(d2, e2, a2, b2, c2, x[3], 14);
    GGG(c2, d2, e2, a2, b2, 0, 14);
    GGG(b2, c2, d2, e2, a2, 0, 6);
    GGG(a2, b2, c2, d2, e2, x[0], 14);
    GGG(e2, a2, b2, c2, d2, x[5], 6);
    GGG(d2, e2, a2, b2, c2, 0, 9);
    GGG(c2, d2, e2, a2, b2, x[2], 12);
    GGG(b2, c2, d2, e2, a2, 0, 9);
    GGG(a2, b2, c2, d2, e2, 0, 12);
    GGG(e2, a2, b2, c2, d2, x[7], 5);
    GGG(d2, e2, a2, b2, c2, 0, 15);
    GGG(c2, d2, e2, a2, b2, x14, 8);

    /* parallel round 5 */
    FFF(b2, c2, d2, e2, a2, 0, 8);
    FFF(a2, b2, c2, d2, e2, 0, 5);
    FFF(e2, a2, b2, c2, d2, 0, 12);
    FFF(d2, e2, a2, b2, c2, x[4], 9);
    FFF(c2, d2, e2, a2, b2, x[1], 12);
    FFF(b2, c2, d2, e2, a2, x[5], 5);
    FFF(a2, b2, c2, d2, e2, x8, 14);
    FFF(e2, a2, b2, c2, d2, x[7], 6);
    FFF(d2, e2, a2, b2, c2, x[6], 8);
    FFF(c2, d2, e2, a2, b2, x[2], 13);
    FFF(b2, c2, d2, e2, a2, 0, 6);
    FFF(a2, b2, c2, d2, e2, x14, 5);
    FFF(e2, a2, b2, c2, d2, x[0], 15);
    FFF(d2, e2, a2, b2, c2, x[3], 13);
    FFF(c2, d2, e2, a2, b2, 0, 11);
    FFF(b2, c2, d2, e2, a2, 0, 11);

    digest[0] = c1 + d2;
    digest[1] = d1 + e2;
    digest[2] = e1 + a2;
    digest[3] = a1 + b2;
    digest[4] = b1 + c2;
}
#endif
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
unsigned int addc(unsigned int a, unsigned int b, unsigned int *carry)
{
    uint64_t sum64 = (uint64_t)a + (uint64_t)b + (uint64_t)*carry;
    unsigned sum = (unsigned int)sum64;
    *carry = (unsigned int)(sum64 >> 32) & 1;

    return sum;
}

// Subtract with borrow
unsigned int subc(unsigned int a, unsigned int b, unsigned int *borrow)
{
    uint64_t diff64 = (uint64_t)a - b - *borrow;
    unsigned int diff = (unsigned int)diff64;
    *borrow = (unsigned int)((diff64 >> 32) & 1);

    return diff;
}

// 32 x 32 multiply-add
void madd(unsigned int *high, unsigned int *low, unsigned int a, unsigned int b, unsigned int c)
{
    uint64_t mul64 = (uint64_t)a * b + c;
    *low = (unsigned int)mul64;
    *high = (unsigned int)(mul64 >> 32);
}


unsigned int sub256(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    unsigned int borrow = 0;
    for(int i = 7; i >= 0; i--) {
        c[i] = subc(a[i], b[i], &borrow);
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
    unsigned int carry = 0;

    for(int i = 7; i >= 0; i--) {
        c[i] = addc(a[i], b[i], &carry);
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

bool equal(const unsigned int a[8], const unsigned int b[8])
{
    for(int i = 0; i < 8; i++) {
        if(a[i] != b[i]) {
            return false;
        }
    }

    return true;
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
void subModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    if(sub256(a, b, c)) {
        addP(c, c);
    }
}


void addModP(const unsigned int a[8], const unsigned int b[8], unsigned int c[8])
{
    unsigned int carry = 0;

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
        product[i + 7] = addc(product[i + 7], high[i], &carry);
    }
    product[6] = addc(product[6], 0, &carry);

    carry = 0;

    // Multiply high by 977 and add to low
    // Affects product[15] to product[5]
    for(int i = 7; i >= 0; i--) {
        unsigned int t = 0;
        madd(&hWord, &t, high[i], 977, hWord);
        product[8 + i] = addc(product[8 + i], t, &carry);
    }
    product[7] = addc(product[7], hWord, &carry);
    product[6] = addc(0, 0, &carry);

    // Multiply high 2 words by 2^32 and add to low
    // Affects product[14] to product[7]
    carry = 0;
    high[7] = product[7];
    high[6] = product[6];

    product[7] = 0;
    product[6] = 0;

    product[14] = addc(product[14], high[7], &carry);
    product[13] = addc(product[13], high[6], &carry);

    // Propagate the carry
    for(int i = 12; i >= 7; i--) {
        product[i] = addc(product[i], 0, &carry);
    }

    // Multiply top 2 words by 977 and add to low
    // Affects product[15] to product[7]
    carry = 0;
    hWord = 0;
    unsigned int t = 0;
    madd(&hWord, &t, high[7], 977, hWord);
    product[15] = addc(product[15], t, &carry);
    madd(&hWord, &t, high[6], 977, hWord);
    product[14] = addc(product[14], t, &carry);
    product[13] = addc(product[13], hWord, &carry);

    // Propagate carry
    for(int i = 12; i >= 7; i--) {
        product[i] = addc(product[i], 0, &carry);
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
    // Strange behavior here: Incorrect results if in a single loop.
    for(int i = 0; i < 19; i++) {
        mulModP_d(x, y);
        squareModP_d(x);
    }
    
    for(int i = 0; i < 1; i++) {
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

#define round(a, b, c, d, e, f, g, h, m, k)\
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

    round(a, b, c, d, e, f, g, h, w[0], _K[0]);
    round(h, a, b, c, d, e, f, g, w[1], _K[1]);
    round(g, h, a, b, c, d, e, f, w[2], _K[2]);
    round(f, g, h, a, b, c, d, e, w[3], _K[3]);
    round(e, f, g, h, a, b, c, d, w[4], _K[4]);
    round(d, e, f, g, h, a, b, c, w[5], _K[5]);
    round(c, d, e, f, g, h, a, b, w[6], _K[6]);
    round(b, c, d, e, f, g, h, a, w[7], _K[7]);
    round(a, b, c, d, e, f, g, h, w[8], _K[8]);
    round(h, a, b, c, d, e, f, g, w[9], _K[9]);
    round(g, h, a, b, c, d, e, f, w[10], _K[10]);
    round(f, g, h, a, b, c, d, e, w[11], _K[11]);
    round(e, f, g, h, a, b, c, d, w[12], _K[12]);
    round(d, e, f, g, h, a, b, c, w[13], _K[13]);
    round(c, d, e, f, g, h, a, b, w[14], _K[14]);
    round(b, c, d, e, f, g, h, a, w[15], _K[15]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[16]);
    round(h, a, b, c, d, e, f, g, w[1], _K[17]);
    round(g, h, a, b, c, d, e, f, w[2], _K[18]);
    round(f, g, h, a, b, c, d, e, w[3], _K[19]);
    round(e, f, g, h, a, b, c, d, w[4], _K[20]);
    round(d, e, f, g, h, a, b, c, w[5], _K[21]);
    round(c, d, e, f, g, h, a, b, w[6], _K[22]);
    round(b, c, d, e, f, g, h, a, w[7], _K[23]);
    round(a, b, c, d, e, f, g, h, w[8], _K[24]);
    round(h, a, b, c, d, e, f, g, w[9], _K[25]);
    round(g, h, a, b, c, d, e, f, w[10], _K[26]);
    round(f, g, h, a, b, c, d, e, w[11], _K[27]);
    round(e, f, g, h, a, b, c, d, w[12], _K[28]);
    round(d, e, f, g, h, a, b, c, w[13], _K[29]);
    round(c, d, e, f, g, h, a, b, w[14], _K[30]);
    round(b, c, d, e, f, g, h, a, w[15], _K[31]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[32]);
    round(h, a, b, c, d, e, f, g, w[1], _K[33]);
    round(g, h, a, b, c, d, e, f, w[2], _K[34]);
    round(f, g, h, a, b, c, d, e, w[3], _K[35]);
    round(e, f, g, h, a, b, c, d, w[4], _K[36]);
    round(d, e, f, g, h, a, b, c, w[5], _K[37]);
    round(c, d, e, f, g, h, a, b, w[6], _K[38]);
    round(b, c, d, e, f, g, h, a, w[7], _K[39]);
    round(a, b, c, d, e, f, g, h, w[8], _K[40]);
    round(h, a, b, c, d, e, f, g, w[9], _K[41]);
    round(g, h, a, b, c, d, e, f, w[10], _K[42]);
    round(f, g, h, a, b, c, d, e, w[11], _K[43]);
    round(e, f, g, h, a, b, c, d, w[12], _K[44]);
    round(d, e, f, g, h, a, b, c, w[13], _K[45]);
    round(c, d, e, f, g, h, a, b, w[14], _K[46]);
    round(b, c, d, e, f, g, h, a, w[15], _K[47]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[48]);
    round(h, a, b, c, d, e, f, g, w[1], _K[49]);
    round(g, h, a, b, c, d, e, f, w[2], _K[50]);
    round(f, g, h, a, b, c, d, e, w[3], _K[51]);
    round(e, f, g, h, a, b, c, d, w[4], _K[52]);
    round(d, e, f, g, h, a, b, c, w[5], _K[53]);
    round(c, d, e, f, g, h, a, b, w[6], _K[54]);
    round(b, c, d, e, f, g, h, a, w[7], _K[55]);
    round(a, b, c, d, e, f, g, h, w[8], _K[56]);
    round(h, a, b, c, d, e, f, g, w[9], _K[57]);
    round(g, h, a, b, c, d, e, f, w[10], _K[58]);
    round(f, g, h, a, b, c, d, e, w[11], _K[59]);
    round(e, f, g, h, a, b, c, d, w[12], _K[60]);
    round(d, e, f, g, h, a, b, c, w[13], _K[61]);
    round(c, d, e, f, g, h, a, b, w[14], _K[62]);
    round(b, c, d, e, f, g, h, a, w[15], _K[63]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[0]);
    round(h, a, b, c, d, e, f, g, 0, _K[1]);
    round(g, h, a, b, c, d, e, f, 0, _K[2]);
    round(f, g, h, a, b, c, d, e, 0, _K[3]);
    round(e, f, g, h, a, b, c, d, 0, _K[4]);
    round(d, e, f, g, h, a, b, c, 0, _K[5]);
    round(c, d, e, f, g, h, a, b, 0, _K[6]);
    round(b, c, d, e, f, g, h, a, 0, _K[7]);
    round(a, b, c, d, e, f, g, h, 0, _K[8]);
    round(h, a, b, c, d, e, f, g, 0, _K[9]);
    round(g, h, a, b, c, d, e, f, 0, _K[10]);
    round(f, g, h, a, b, c, d, e, 0, _K[11]);
    round(e, f, g, h, a, b, c, d, 0, _K[12]);
    round(d, e, f, g, h, a, b, c, 0, _K[13]);
    round(c, d, e, f, g, h, a, b, 0, _K[14]);
    round(b, c, d, e, f, g, h, a, w[15], _K[15]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[16]);
    round(h, a, b, c, d, e, f, g, w[1], _K[17]);
    round(g, h, a, b, c, d, e, f, w[2], _K[18]);
    round(f, g, h, a, b, c, d, e, w[3], _K[19]);
    round(e, f, g, h, a, b, c, d, w[4], _K[20]);
    round(d, e, f, g, h, a, b, c, w[5], _K[21]);
    round(c, d, e, f, g, h, a, b, w[6], _K[22]);
    round(b, c, d, e, f, g, h, a, w[7], _K[23]);
    round(a, b, c, d, e, f, g, h, w[8], _K[24]);
    round(h, a, b, c, d, e, f, g, w[9], _K[25]);
    round(g, h, a, b, c, d, e, f, w[10], _K[26]);
    round(f, g, h, a, b, c, d, e, w[11], _K[27]);
    round(e, f, g, h, a, b, c, d, w[12], _K[28]);
    round(d, e, f, g, h, a, b, c, w[13], _K[29]);
    round(c, d, e, f, g, h, a, b, w[14], _K[30]);
    round(b, c, d, e, f, g, h, a, w[15], _K[31]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[32]);
    round(h, a, b, c, d, e, f, g, w[1], _K[33]);
    round(g, h, a, b, c, d, e, f, w[2], _K[34]);
    round(f, g, h, a, b, c, d, e, w[3], _K[35]);
    round(e, f, g, h, a, b, c, d, w[4], _K[36]);
    round(d, e, f, g, h, a, b, c, w[5], _K[37]);
    round(c, d, e, f, g, h, a, b, w[6], _K[38]);
    round(b, c, d, e, f, g, h, a, w[7], _K[39]);
    round(a, b, c, d, e, f, g, h, w[8], _K[40]);
    round(h, a, b, c, d, e, f, g, w[9], _K[41]);
    round(g, h, a, b, c, d, e, f, w[10], _K[42]);
    round(f, g, h, a, b, c, d, e, w[11], _K[43]);
    round(e, f, g, h, a, b, c, d, w[12], _K[44]);
    round(d, e, f, g, h, a, b, c, w[13], _K[45]);
    round(c, d, e, f, g, h, a, b, w[14], _K[46]);
    round(b, c, d, e, f, g, h, a, w[15], _K[47]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[48]);
    round(h, a, b, c, d, e, f, g, w[1], _K[49]);
    round(g, h, a, b, c, d, e, f, w[2], _K[50]);
    round(f, g, h, a, b, c, d, e, w[3], _K[51]);
    round(e, f, g, h, a, b, c, d, w[4], _K[52]);
    round(d, e, f, g, h, a, b, c, w[5], _K[53]);
    round(c, d, e, f, g, h, a, b, w[6], _K[54]);
    round(b, c, d, e, f, g, h, a, w[7], _K[55]);
    round(a, b, c, d, e, f, g, h, w[8], _K[56]);
    round(h, a, b, c, d, e, f, g, w[9], _K[57]);
    round(g, h, a, b, c, d, e, f, w[10], _K[58]);
    round(f, g, h, a, b, c, d, e, w[11], _K[59]);
    round(e, f, g, h, a, b, c, d, w[12], _K[60]);
    round(d, e, f, g, h, a, b, c, w[13], _K[61]);
    round(c, d, e, f, g, h, a, b, w[14], _K[62]);
    round(b, c, d, e, f, g, h, a, w[15], _K[63]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[0]);
    round(h, a, b, c, d, e, f, g, w[1], _K[1]);
    round(g, h, a, b, c, d, e, f, w[2], _K[2]);
    round(f, g, h, a, b, c, d, e, w[3], _K[3]);
    round(e, f, g, h, a, b, c, d, w[4], _K[4]);
    round(d, e, f, g, h, a, b, c, w[5], _K[5]);
    round(c, d, e, f, g, h, a, b, w[6], _K[6]);
    round(b, c, d, e, f, g, h, a, w[7], _K[7]);
    round(a, b, c, d, e, f, g, h, w[8], _K[8]);
    round(h, a, b, c, d, e, f, g, 0, _K[9]);
    round(g, h, a, b, c, d, e, f, 0, _K[10]);
    round(f, g, h, a, b, c, d, e, 0, _K[11]);
    round(e, f, g, h, a, b, c, d, 0, _K[12]);
    round(d, e, f, g, h, a, b, c, 0, _K[13]);
    round(c, d, e, f, g, h, a, b, 0, _K[14]);
    round(b, c, d, e, f, g, h, a, w[15], _K[15]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[16]);
    round(h, a, b, c, d, e, f, g, w[1], _K[17]);
    round(g, h, a, b, c, d, e, f, w[2], _K[18]);
    round(f, g, h, a, b, c, d, e, w[3], _K[19]);
    round(e, f, g, h, a, b, c, d, w[4], _K[20]);
    round(d, e, f, g, h, a, b, c, w[5], _K[21]);
    round(c, d, e, f, g, h, a, b, w[6], _K[22]);
    round(b, c, d, e, f, g, h, a, w[7], _K[23]);
    round(a, b, c, d, e, f, g, h, w[8], _K[24]);
    round(h, a, b, c, d, e, f, g, w[9], _K[25]);
    round(g, h, a, b, c, d, e, f, w[10], _K[26]);
    round(f, g, h, a, b, c, d, e, w[11], _K[27]);
    round(e, f, g, h, a, b, c, d, w[12], _K[28]);
    round(d, e, f, g, h, a, b, c, w[13], _K[29]);
    round(c, d, e, f, g, h, a, b, w[14], _K[30]);
    round(b, c, d, e, f, g, h, a, w[15], _K[31]);

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

    round(a, b, c, d, e, f, g, h, w[0], _K[32]);
    round(h, a, b, c, d, e, f, g, w[1], _K[33]);
    round(g, h, a, b, c, d, e, f, w[2], _K[34]);
    round(f, g, h, a, b, c, d, e, w[3], _K[35]);
    round(e, f, g, h, a, b, c, d, w[4], _K[36]);
    round(d, e, f, g, h, a, b, c, w[5], _K[37]);
    round(c, d, e, f, g, h, a, b, w[6], _K[38]);
    round(b, c, d, e, f, g, h, a, w[7], _K[39]);
    round(a, b, c, d, e, f, g, h, w[8], _K[40]);
    round(h, a, b, c, d, e, f, g, w[9], _K[41]);
    round(g, h, a, b, c, d, e, f, w[10], _K[42]);
    round(f, g, h, a, b, c, d, e, w[11], _K[43]);
    round(e, f, g, h, a, b, c, d, w[12], _K[44]);
    round(d, e, f, g, h, a, b, c, w[13], _K[45]);
    round(c, d, e, f, g, h, a, b, w[14], _K[46]);
    round(b, c, d, e, f, g, h, a, w[15], _K[47]);


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

    round(a, b, c, d, e, f, g, h, w[0], _K[48]);
    round(h, a, b, c, d, e, f, g, w[1], _K[49]);
    round(g, h, a, b, c, d, e, f, w[2], _K[50]);
    round(f, g, h, a, b, c, d, e, w[3], _K[51]);
    round(e, f, g, h, a, b, c, d, w[4], _K[52]);
    round(d, e, f, g, h, a, b, c, w[5], _K[53]);
    round(c, d, e, f, g, h, a, b, w[6], _K[54]);
    round(b, c, d, e, f, g, h, a, w[7], _K[55]);
    round(a, b, c, d, e, f, g, h, w[8], _K[56]);
    round(h, a, b, c, d, e, f, g, w[9], _K[57]);
    round(g, h, a, b, c, d, e, f, w[10], _K[58]);
    round(f, g, h, a, b, c, d, e, w[11], _K[59]);
    round(e, f, g, h, a, b, c, d, w[12], _K[60]);
    round(d, e, f, g, h, a, b, c, w[13], _K[61]);
    round(c, d, e, f, g, h, a, b, w[14], _K[62]);
    round(b, c, d, e, f, g, h, a, w[15], _K[63]);

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
    int thread;
    int block;
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
}CLDeviceResult;

bool isInList(unsigned int hash[5], __global unsigned int *targetList, size_t numTargets)
{
    bool found = false;

    for(size_t i = 0; i < numTargets; i++) {
        int equal = 0;

        for(int j = 0; j < 5; j++) {
            if(hash[j] == targetList[5 * i + j]) {
                equal++;
            }
        }

        if(equal == 5) {
            found = true;
        }
    }

    return found;
}

bool isInBloomFilter(unsigned int hash[5], __global unsigned int *targetList, ulong mask)
{
    bool foundMatch = true;

    unsigned int h5 = 0;

    for(int i = 0; i < 5; i++) {
        h5 += hash[i];
    }

    uint64_t idx[5];

    idx[0] = ((hash[0] << 6) | (h5 & 0x3f)) & mask;
    idx[1] = ((hash[1] << 6) | ((h5 >> 6) & 0x3f)) & mask;
    idx[2] = ((hash[2] << 6) | ((h5 >> 12) & 0x3f)) & mask;
    idx[3] = ((hash[3] << 6) | ((h5 >> 18) & 0x3f)) & mask;
    idx[4] = ((hash[4] << 6) | ((h5 >> 24) & 0x3f)) & mask;

    for(int i = 0; i < 5; i++) {
        unsigned int j = idx[i];
        unsigned int f = targetList[j / 32];

        if((f & (0x01 << (j % 32))) == 0) {
            foundMatch = false;
        }
    }

    return foundMatch;
}

bool checkHash(unsigned int hash[5], __global unsigned int *targetList, size_t numTargets, ulong mask)
{
    if(numTargets > 16) {
        return isInBloomFilter(hash, targetList, mask);
    } else {
        return isInList(hash, targetList, numTargets);
    }
}


void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    const unsigned int iv[5] = {
        0x67452301,
        0xefcdab89,
        0x98badcfe,
        0x10325476,
        0xc3d2e1f0
    };

    for(int i = 0; i < 5; i++) {
        hOut[i] = endian(hIn[i] + iv[(i + 1) % 5]);
    }
}


__kernel void multiplyStepKernel(
    int pointsPerThread,
    int step,
    __global unsigned int *privateKeys,
    __global unsigned int *chain,
    __global unsigned int *gxPtr,
    __global unsigned int *gyPtr,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr)
{
    unsigned int gx[8];
    unsigned int gy[8];

    for(int i = 0; i < 8; i++) {
        gx[i] = gxPtr[step * 8 + i];
        gy[i] = gyPtr[step * 8 + i];
    }

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};

    int batchIdx = 0;
    for(int i = 0; i < pointsPerThread; i++) {

        unsigned int p[8];
        readInt(privateKeys, i, p);

        unsigned int bit = p[7 - step / 32] & (1 << (step % 32));


        unsigned int x[8];
        readInt(xPtr, i, x);
        

        if(bit != 0) {
            if(!isInfinity(x)) {
                beginBatchAddWithDouble(gx, gy, xPtr, chain, i, batchIdx, inverse);
                batchIdx++;
            }
        }
    }

    doBatchInverse(inverse);


    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        unsigned int p[8];
        readInt(privateKeys, i, p);
        unsigned int bit = p[7 - step / 32] & (1 << (step % 32));

        unsigned int x[8];
        readInt(xPtr, i, x);

        bool infinity = isInfinity(x);

        if(bit != 0) {
            if(!infinity) {
                batchIdx--;
                completeBatchAddWithDouble(gx, gy, xPtr, yPtr, i, batchIdx, chain, inverse, newX, newY);
            } else {
                copyBigInt(gx, newX);
                copyBigInt(gy, newY);
            }

            writeInt(xPtr, i, newX);
            writeInt(yPtr, i, newY);

        }
    }
}


void hashPublicKey(const unsigned int *x, const unsigned int *y, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKey(x, y, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

void hashPublicKeyCompressed(const unsigned int *x, unsigned int yParity, unsigned int *digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x, yParity, hash);

    // Swap to little-endian
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);

}

void atomicListAdd(__global CLDeviceResult *results, __global unsigned int *numResults, CLDeviceResult *r)
{
    unsigned int count = atomic_add(numResults, 1);

    results[count] = *r;
}

void setResultFound(int idx, bool compressed, unsigned int x[8], unsigned int y[8], unsigned int digest[5], __global CLDeviceResult *results, __global unsigned int *numResults)
{
    CLDeviceResult r;

    r.block = get_group_id(0);
    r.thread = get_local_id(0);
    r.idx = idx;
    r.compressed = compressed;

    for(int i = 0; i < 8; i++) {
        r.x[i] = x[i];
        r.y[i] = y[i];
    }

    doRMD160FinalRound(digest, r.digest);

    atomicListAdd(results, numResults, &r);
}

void doIteration(
    size_t pointsPerThread,
    int compression,
    __global unsigned int *chain,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr,
    __global unsigned int *incXPtr,
    __global unsigned int *incYPtr,
    __global unsigned int *targetList,
    size_t numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    unsigned int incX[8];
    unsigned int incY[8];

    for(int i = 0; i < 8; i++) {
        incX[i] = incXPtr[i];
        incY[i] = incYPtr[i];
    }

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];

        unsigned int digest[5];

        readInt(xPtr, i, x);

        if((compression == UNCOMPRESSED) || (compression == BOTH)) {
            unsigned int y[8];
            readInt(yPtr, i, y);

            hashPublicKey(x, y, digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW(yPtr, i), digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
                unsigned int y[8];
                readInt(yPtr, i, y);
                setResultFound(i, true, x, y, digest, results, numResults);
            }
        }

        beginBatchAdd(incX, x, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAdd(incX, incY, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}


void doIterationWithDouble(
    size_t pointsPerThread,
    int compression,
    __global unsigned int *chain,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr,
    __global unsigned int *incXPtr,
    __global unsigned int *incYPtr,
    __global unsigned int *targetList,
    size_t numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    unsigned int incX[8];
    unsigned int incY[8];

    for(int i = 0; i < 8; i++) {
        incX[i] = incXPtr[i];
        incY[i] = incYPtr[i];
    }

    // Multiply together all (_Gx - x) and then invert
    unsigned int inverse[8] = {0,0,0,0,0,0,0,1};
    for(int i = 0; i < pointsPerThread; i++) {
        unsigned int x[8];

        unsigned int digest[5];

        readInt(xPtr, i, x);

        // uncompressed
        if((compression == UNCOMPRESSED) || (compression == BOTH)) {
            unsigned int y[8];
            readInt(yPtr, i, y);
            hashPublicKey(x, y, digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        // compressed
        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW(yPtr, i), digest);

            if(checkHash(digest, targetList, numTargets, mask)) {

                unsigned int y[8];
                readInt(yPtr, i, y);

                setResultFound(i, true, x, y, digest, results, numResults);
            }
        }

        beginBatchAddWithDouble(incX, incY, xPtr, chain, i, i, inverse);
    }

    doBatchInverse(inverse);

    for(int i = pointsPerThread - 1; i >= 0; i--) {

        unsigned int newX[8];
        unsigned int newY[8];

        completeBatchAddWithDouble(incX, incY, xPtr, yPtr, i, i, chain, inverse, newX, newY);

        writeInt(xPtr, i, newX);
        writeInt(yPtr, i, newY);
    }
}

/**
* Performs a single iteration
*/
__kernel void keyFinderKernel(
    unsigned int pointsPerThread,
    int compression,
    __global unsigned int *chain,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr,
    __global unsigned int *incXPtr,
    __global unsigned int *incYPtr,
    __global unsigned int *targetList,
    ulong numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    doIteration(pointsPerThread, compression, chain, xPtr, yPtr, incXPtr, incYPtr, targetList, numTargets, mask, results, numResults);
}

__kernel void keyFinderKernelWithDouble(
    unsigned int pointsPerThread,
    int compression,
    __global unsigned int *chain,
    __global unsigned int *xPtr,
    __global unsigned int *yPtr,
    __global unsigned int *incXPtr,
    __global unsigned int *incYPtr,
    __global unsigned int *targetList,
    ulong numTargets,
    ulong mask,
    __global CLDeviceResult *results,
    __global unsigned int *numResults)
{
    doIterationWithDouble(pointsPerThread, compression, chain, xPtr, yPtr, incXPtr, incYPtr, targetList, numTargets, mask, results, numResults);
}
