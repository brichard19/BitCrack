#include"CryptoUtil.h"
#include<stdio.h>

static const unsigned int _IV[5] = {
	0x67452301,
	0xefcdab89,
	0x98badcfe,
	0x10325476,
	0xc3d2e1f0
};

static const unsigned int _K0 = 0x5a827999;
static const unsigned int _K1 = 0x6ed9eba1;
static const unsigned int _K2 = 0x8f1bbcdc;
static const unsigned int _K3 = 0xa953fd4e;

static const unsigned int _K4 = 0x7a6d76e9;
static const unsigned int _K5 = 0x6d703ef3;
static const unsigned int _K6 = 0x5c4dd124;
static const unsigned int _K7 = 0x50a28be6;


static unsigned int endian(unsigned int x)
{
	return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

static unsigned int rotl(unsigned int x, int n)
{
	return (x << n) | (x >> (32 - n));
}

static unsigned int F(unsigned int x, unsigned int y, unsigned int z)
{
	return x ^ y ^ z;
}

static unsigned int G(unsigned int x, unsigned int y, unsigned int z)
{
	return (((x) & (y)) | (~(x) & (z)));
}

static unsigned int H(unsigned int x, unsigned int y, unsigned int z)
{
	return (((x) | ~(y)) ^ (z));
}

static unsigned int I(unsigned int x, unsigned int y, unsigned int z)
{
	return (((x) & (z)) | ((y) & ~(z)));
}

static unsigned int J(unsigned int x, unsigned int y, unsigned int z)
{
	return  ((x) ^ ((y) | ~(z)));
}

static void FF(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += F(b, c, d) + x;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

static void GG(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += G(b, c, d) + x + _K0;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

static void HH(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += H(b, c, d) + x + _K1;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

static void II(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += I(b, c, d) + x + _K2;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

static void JJ(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += J(b, c, d) + x + _K3;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

static void FFF(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += F(b, c, d) + x;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

static void GGG(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += G(b, c, d) + x + _K4;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

static void HHH(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += H(b, c, d) + x + _K5;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

static void III(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += I(b, c, d) + x + _K6;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

static void JJJ(unsigned int &a, unsigned int &b, unsigned int &c, unsigned int &d, unsigned int &e, unsigned int x, unsigned int s)
{
	a += J(b, c, d) + x + _K7;
	a = rotl(a, s) + e;
	c = rotl(c, 10);
}

void crypto::ripemd160(unsigned int *x, unsigned int *digest)
{
	unsigned int a1 = _IV[0];
	unsigned int b1 = _IV[1];
	unsigned int c1 = _IV[2];
	unsigned int d1 = _IV[3];
	unsigned int e1 = _IV[4];

	unsigned int a2 = _IV[0];
	unsigned int b2 = _IV[1];
	unsigned int c2 = _IV[2];
	unsigned int d2 = _IV[3];
	unsigned int e2 = _IV[4];

	/* round 1 */
	FF(a1, b1, c1, d1, e1, x[0], 11);
	FF(e1, a1, b1, c1, d1, x[1], 14);
	FF(d1, e1, a1, b1, c1, x[2], 15);
	FF(c1, d1, e1, a1, b1, x[3], 12);
	FF(b1, c1, d1, e1, a1, x[4], 5);
	FF(a1, b1, c1, d1, e1, x[5], 8);
	FF(e1, a1, b1, c1, d1, x[6], 7);
	FF(d1, e1, a1, b1, c1, x[7], 9);
	FF(c1, d1, e1, a1, b1, x[8], 11);
	FF(b1, c1, d1, e1, a1, x[9], 13);
	FF(a1, b1, c1, d1, e1, x[10], 14);
	FF(e1, a1, b1, c1, d1, x[11], 15);
	FF(d1, e1, a1, b1, c1, x[12], 6);
	FF(c1, d1, e1, a1, b1, x[13], 7);
	FF(b1, c1, d1, e1, a1, x[14], 9);
	FF(a1, b1, c1, d1, e1, x[15], 8);

	/* round 2 */
	GG(e1, a1, b1, c1, d1, x[7], 7);
	GG(d1, e1, a1, b1, c1, x[4], 6);
	GG(c1, d1, e1, a1, b1, x[13], 8);
	GG(b1, c1, d1, e1, a1, x[1], 13);
	GG(a1, b1, c1, d1, e1, x[10], 11);
	GG(e1, a1, b1, c1, d1, x[6], 9);
	GG(d1, e1, a1, b1, c1, x[15], 7);
	GG(c1, d1, e1, a1, b1, x[3], 15);
	GG(b1, c1, d1, e1, a1, x[12], 7);
	GG(a1, b1, c1, d1, e1, x[0], 12);
	GG(e1, a1, b1, c1, d1, x[9], 15);
	GG(d1, e1, a1, b1, c1, x[5], 9);
	GG(c1, d1, e1, a1, b1, x[2], 11);
	GG(b1, c1, d1, e1, a1, x[14], 7);
	GG(a1, b1, c1, d1, e1, x[11], 13);
	GG(e1, a1, b1, c1, d1, x[8], 12);

	/* round 3 */
	HH(d1, e1, a1, b1, c1, x[3], 11);
	HH(c1, d1, e1, a1, b1, x[10], 13);
	HH(b1, c1, d1, e1, a1, x[14], 6);
	HH(a1, b1, c1, d1, e1, x[4], 7);
	HH(e1, a1, b1, c1, d1, x[9], 14);
	HH(d1, e1, a1, b1, c1, x[15], 9);
	HH(c1, d1, e1, a1, b1, x[8], 13);
	HH(b1, c1, d1, e1, a1, x[1], 15);
	HH(a1, b1, c1, d1, e1, x[2], 14);
	HH(e1, a1, b1, c1, d1, x[7], 8);
	HH(d1, e1, a1, b1, c1, x[0], 13);
	HH(c1, d1, e1, a1, b1, x[6], 6);
	HH(b1, c1, d1, e1, a1, x[13], 5);
	HH(a1, b1, c1, d1, e1, x[11], 12);
	HH(e1, a1, b1, c1, d1, x[5], 7);
	HH(d1, e1, a1, b1, c1, x[12], 5);

	/* round 4 */
	II(c1, d1, e1, a1, b1, x[1], 11);
	II(b1, c1, d1, e1, a1, x[9], 12);
	II(a1, b1, c1, d1, e1, x[11], 14);
	II(e1, a1, b1, c1, d1, x[10], 15);
	II(d1, e1, a1, b1, c1, x[0], 14);
	II(c1, d1, e1, a1, b1, x[8], 15);
	II(b1, c1, d1, e1, a1, x[12], 9);
	II(a1, b1, c1, d1, e1, x[4], 8);
	II(e1, a1, b1, c1, d1, x[13], 9);
	II(d1, e1, a1, b1, c1, x[3], 14);
	II(c1, d1, e1, a1, b1, x[7], 5);
	II(b1, c1, d1, e1, a1, x[15], 6);
	II(a1, b1, c1, d1, e1, x[14], 8);
	II(e1, a1, b1, c1, d1, x[5], 6);
	II(d1, e1, a1, b1, c1, x[6], 5);
	II(c1, d1, e1, a1, b1, x[2], 12);

	/* round 5 */
	JJ(b1, c1, d1, e1, a1, x[4], 9);
	JJ(a1, b1, c1, d1, e1, x[0], 15);
	JJ(e1, a1, b1, c1, d1, x[5], 5);
	JJ(d1, e1, a1, b1, c1, x[9], 11);
	JJ(c1, d1, e1, a1, b1, x[7], 6);
	JJ(b1, c1, d1, e1, a1, x[12], 8);
	JJ(a1, b1, c1, d1, e1, x[2], 13);
	JJ(e1, a1, b1, c1, d1, x[10], 12);
	JJ(d1, e1, a1, b1, c1, x[14], 5);
	JJ(c1, d1, e1, a1, b1, x[1], 12);
	JJ(b1, c1, d1, e1, a1, x[3], 13);
	JJ(a1, b1, c1, d1, e1, x[8], 14);
	JJ(e1, a1, b1, c1, d1, x[11], 11);
	JJ(d1, e1, a1, b1, c1, x[6], 8);
	JJ(c1, d1, e1, a1, b1, x[15], 5);
	JJ(b1, c1, d1, e1, a1, x[13], 6);


	/* parallel round 1 */
	JJJ(a2, b2, c2, d2, e2, x[5], 8);
	JJJ(e2, a2, b2, c2, d2, x[14], 9);
	JJJ(d2, e2, a2, b2, c2, x[7], 9);
	JJJ(c2, d2, e2, a2, b2, x[0], 11);
	JJJ(b2, c2, d2, e2, a2, x[9], 13);
	JJJ(a2, b2, c2, d2, e2, x[2], 15);
	JJJ(e2, a2, b2, c2, d2, x[11], 15);
	JJJ(d2, e2, a2, b2, c2, x[4], 5);
	JJJ(c2, d2, e2, a2, b2, x[13], 7);
	JJJ(b2, c2, d2, e2, a2, x[6], 7);
	JJJ(a2, b2, c2, d2, e2, x[15], 8);
	JJJ(e2, a2, b2, c2, d2, x[8], 11);
	JJJ(d2, e2, a2, b2, c2, x[1], 14);
	JJJ(c2, d2, e2, a2, b2, x[10], 14);
	JJJ(b2, c2, d2, e2, a2, x[3], 12);
	JJJ(a2, b2, c2, d2, e2, x[12], 6);

	/* parallel round 2 */
	III(e2, a2, b2, c2, d2, x[6], 9);
	III(d2, e2, a2, b2, c2, x[11], 13);
	III(c2, d2, e2, a2, b2, x[3], 15);
	III(b2, c2, d2, e2, a2, x[7], 7);
	III(a2, b2, c2, d2, e2, x[0], 12);
	III(e2, a2, b2, c2, d2, x[13], 8);
	III(d2, e2, a2, b2, c2, x[5], 9);
	III(c2, d2, e2, a2, b2, x[10], 11);
	III(b2, c2, d2, e2, a2, x[14], 7);
	III(a2, b2, c2, d2, e2, x[15], 7);
	III(e2, a2, b2, c2, d2, x[8], 12);
	III(d2, e2, a2, b2, c2, x[12], 7);
	III(c2, d2, e2, a2, b2, x[4], 6);
	III(b2, c2, d2, e2, a2, x[9], 15);
	III(a2, b2, c2, d2, e2, x[1], 13);
	III(e2, a2, b2, c2, d2, x[2], 11);

	/* parallel round 3 */
	HHH(d2, e2, a2, b2, c2, x[15], 9);
	HHH(c2, d2, e2, a2, b2, x[5], 7);
	HHH(b2, c2, d2, e2, a2, x[1], 15);
	HHH(a2, b2, c2, d2, e2, x[3], 11);
	HHH(e2, a2, b2, c2, d2, x[7], 8);
	HHH(d2, e2, a2, b2, c2, x[14], 6);
	HHH(c2, d2, e2, a2, b2, x[6], 6);
	HHH(b2, c2, d2, e2, a2, x[9], 14);
	HHH(a2, b2, c2, d2, e2, x[11], 12);
	HHH(e2, a2, b2, c2, d2, x[8], 13);
	HHH(d2, e2, a2, b2, c2, x[12], 5);
	HHH(c2, d2, e2, a2, b2, x[2], 14);
	HHH(b2, c2, d2, e2, a2, x[10], 13);
	HHH(a2, b2, c2, d2, e2, x[0], 13);
	HHH(e2, a2, b2, c2, d2, x[4], 7);
	HHH(d2, e2, a2, b2, c2, x[13], 5);

	/* parallel round 4 */
	GGG(c2, d2, e2, a2, b2, x[8], 15);
	GGG(b2, c2, d2, e2, a2, x[6], 5);
	GGG(a2, b2, c2, d2, e2, x[4], 8);
	GGG(e2, a2, b2, c2, d2, x[1], 11);
	GGG(d2, e2, a2, b2, c2, x[3], 14);
	GGG(c2, d2, e2, a2, b2, x[11], 14);
	GGG(b2, c2, d2, e2, a2, x[15], 6);
	GGG(a2, b2, c2, d2, e2, x[0], 14);
	GGG(e2, a2, b2, c2, d2, x[5], 6);
	GGG(d2, e2, a2, b2, c2, x[12], 9);
	GGG(c2, d2, e2, a2, b2, x[2], 12);
	GGG(b2, c2, d2, e2, a2, x[13], 9);
	GGG(a2, b2, c2, d2, e2, x[9], 12);
	GGG(e2, a2, b2, c2, d2, x[7], 5);
	GGG(d2, e2, a2, b2, c2, x[10], 15);
	GGG(c2, d2, e2, a2, b2, x[14], 8);

	/* parallel round 5 */
	FFF(b2, c2, d2, e2, a2, x[12], 8);
	FFF(a2, b2, c2, d2, e2, x[15], 5);
	FFF(e2, a2, b2, c2, d2, x[10], 12);
	FFF(d2, e2, a2, b2, c2, x[4], 9);
	FFF(c2, d2, e2, a2, b2, x[1], 12);
	FFF(b2, c2, d2, e2, a2, x[5], 5);
	FFF(a2, b2, c2, d2, e2, x[8], 14);
	FFF(e2, a2, b2, c2, d2, x[7], 6);
	FFF(d2, e2, a2, b2, c2, x[6], 8);
	FFF(c2, d2, e2, a2, b2, x[2], 13);
	FFF(b2, c2, d2, e2, a2, x[13], 6);
	FFF(a2, b2, c2, d2, e2, x[14], 5);
	FFF(e2, a2, b2, c2, d2, x[0], 15);
	FFF(d2, e2, a2, b2, c2, x[3], 13);
	FFF(c2, d2, e2, a2, b2, x[9], 11);
	FFF(b2, c2, d2, e2, a2, x[11], 11);

	digest[0] = endian(_IV[1] + c1 + d2);
	digest[1] = endian(_IV[2] + d1 + e2);
	digest[2] = endian(_IV[3] + e1 + a2);
	digest[3] = endian(_IV[4] + a1 + b2);
	digest[4] = endian(_IV[0] + b1 + c2);
}