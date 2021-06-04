#ifndef RIPEMD160_CL
#define RIPEMD160_CL

#ifndef endian
#define endian(x) ((x) << 24) | (((x) << 8) & 0x00ff0000) | (((x) >> 8) & 0x0000ff00) | ((x) >> 24)
#endif

__constant unsigned int RIPEMD160_IV[5] = {
    0x67452301, 0xefcdab89, 0x98badcfe, 0x10325476, 0xc3d2e1f0,
};

__constant unsigned int K[8] = {
    0x5a827999, 0x6ed9eba1, 0x8f1bbcdc, 0xa953fd4e, 0x7a6d76e9, 0x6d703ef3, 0x5c4dd124, 0x50a28be6
};

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
    a += G((b), (c), (d)) + (x) + K[0];\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define HH(a, b, c, d, e, x, s)\
    a += H((b), (c), (d)) + (x) + K[1];\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define II(a, b, c, d, e, x, s)\
    a += I((b), (c), (d)) + (x) + K[2];\
    a = rotl((a), (s)) + e;\
    c = rotl((c), 10)

#define JJ(a, b, c, d, e, x, s)\
    a += J((b), (c), (d)) + (x) + K[3];\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define FFF(a, b, c, d, e, x, s)\
    a += F((b), (c), (d)) + (x);\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define GGG(a, b, c, d, e, x, s)\
    a += G((b), (c), (d)) + x + K[4];\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define HHH(a, b, c, d, e, x, s)\
    a += H((b), (c), (d)) + (x) + K[5];\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define III(a, b, c, d, e, x, s)\
    a += I((b), (c), (d)) + (x) + K[6];\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

#define JJJ(a, b, c, d, e, x, s)\
    a += J((b), (c), (d)) + (x) + K[7];\
    a = rotl((a), (s)) + (e);\
    c = rotl((c), 10)

void ripemd160p1(const unsigned int x[8], unsigned int digest[5])
{
    unsigned int a = RIPEMD160_IV[0];
    unsigned int b = RIPEMD160_IV[1];
    unsigned int c = RIPEMD160_IV[2];
    unsigned int d = RIPEMD160_IV[3];
    unsigned int e = RIPEMD160_IV[4];

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
    unsigned int a = RIPEMD160_IV[0];
    unsigned int b = RIPEMD160_IV[1];
    unsigned int c = RIPEMD160_IV[2];
    unsigned int d = RIPEMD160_IV[3];
    unsigned int e = RIPEMD160_IV[4];

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

void doRMD160FinalRound(const unsigned int hIn[5], unsigned int hOut[5])
{
    hOut[0] = endian(hIn[0] + 0xefcdab89);
    hOut[1] = endian(hIn[1] + 0x98badcfe);
    hOut[2] = endian(hIn[2] + 0x10325476);
    hOut[3] = endian(hIn[3] + 0xc3d2e1f0);
    hOut[4] = endian(hIn[4] + 0x67452301);
}

#endif
