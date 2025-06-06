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
            if(!isInfinity256k(x)) {
                beginBatchAddWithDouble256k(gx, gy, xPtr, chain, i, batchIdx, &inverse);
                batchIdx++;
            }
        }
    }

    //doBatchInverse(inverse);
    inverse = doBatchInverse256k(inverse);

    i -= dim;
    for(; i >= 0; i -= dim) {
        uint256_t newX;
        uint256_t newY;

        unsigned int p;
        p = readWord256k(privateKeys, i, 7 - step / 32);
        unsigned int bit = p & (1 << (step % 32));

        uint256_t x = xPtr[i];
        bool infinity = isInfinity256k(x);

        if(bit != 0) {
            if(!infinity) {
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
    for(int i = 0; i < 8; i++) {
        hash[i] = endian(hash[i]);
    }

    ripemd160sha256NoFinal(hash, digestOut);
}

void hashPublicKeyCompressed(uint256_t x, unsigned int yParity, unsigned int* digestOut)
{
    unsigned int hash[8];

    sha256PublicKeyCompressed(x.v, yParity, hash);

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

void setResultFound(int idx, bool compressed, uint256_t x, uint256_t y, unsigned int digest[5], __global CLDeviceResult* results, __global unsigned int* numResults)
{
    CLDeviceResult r;

    r.idx = idx;
    r.compressed = compressed;

    for(int i = 0; i < 8; i++) {
        r.x[i] = x.v[i];
        r.y[i] = y.v[i];
    }

    doRMD160FinalRound(digest, r.digest);

    atomicListAdd(results, numResults, &r);
}

void doIteration(
    size_t totalPoints,
    int compression,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int *targetList,
    size_t numTargets,
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

            if(checkHash(digest, targetList, numTargets, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW256k(yPtr, i), digest);

            if(checkHash(digest, targetList, numTargets, mask)) {
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


void doIterationWithDouble(
    size_t totalPoints,
    int compression,
    __global uint256_t* chain,
    __global uint256_t* xPtr,
    __global uint256_t* yPtr,
    __global uint256_t* incXPtr,
    __global uint256_t* incYPtr,
    __global unsigned int* targetList,
    size_t numTargets,
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

            if(checkHash(digest, targetList, numTargets, mask)) {
                setResultFound(i, false, x, y, digest, results, numResults);
            }
        }

        // compressed
        if((compression == COMPRESSED) || (compression == BOTH)) {

            hashPublicKeyCompressed(x, readLSW256k(yPtr, i), digest);

            if(checkHash(digest, targetList, numTargets, mask)) {

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

/**
* Performs a single iteration
*/
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
    doIteration(totalPoints, compression, chain, xPtr, yPtr, incXPtr, incYPtr, targetList, numTargets, mask, results, numResults);
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
    doIterationWithDouble(totalPoints, compression, chain, xPtr, yPtr, incXPtr, incYPtr, targetList, numTargets, mask, results, numResults);
}
