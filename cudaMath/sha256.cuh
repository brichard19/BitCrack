#ifndef _SHA256_CUH
#define _SHA256_CUH

#include<cuda.h>
#include<cuda_runtime.h>


__constant__ unsigned int _K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__constant__ unsigned int _IV[8] = {
	0x6a09e667,
	0xbb67ae85,
	0x3c6ef372,
	0xa54ff53a,
	0x510e527f,
	0x9b05688c,
	0x1f83d9ab,
	0x5be0cd19
};


__device__ __forceinline__ unsigned int rotr(unsigned int x, int n)
{
	return (x >> n) ^ (x << (32 - n));
}

__device__ __forceinline__ unsigned int MAJ(unsigned int a, unsigned int b, unsigned int c)
{
	return (a & b) ^ (a & c) ^ (b & c);
}

__device__ __forceinline__ unsigned int CH(unsigned int e, unsigned int f, unsigned int g)
{
	return (e & f) ^ (~e & g);
}

__device__ __forceinline__ unsigned int s0(unsigned int x)
{
	return rotr(x, 7) ^ rotr(x, 18) ^ (x >> 3);
}

__device__ __forceinline__ unsigned int s1(unsigned int x)
{
	return rotr(x, 17) ^ rotr(x, 19) ^ (x >> 10);
}


__device__ __forceinline__ void roundSha256(unsigned int a, unsigned int b, unsigned int c, unsigned int &d, unsigned int e, unsigned int f, unsigned int g, unsigned int &h, unsigned int m, unsigned int k)
{
	unsigned int s = CH(e, f, g) + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + k + m;

	d += s + h;

	h += s + MAJ(a, b, c) + (rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22));
}

__device__ void sha256PublicKey(const unsigned int x[8], const unsigned int y[8], unsigned int digest[8])
{
	unsigned int a, b, c, d, e, f, g, h;
	unsigned int w[16];

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


	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[0]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[1]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[2]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[3]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[4]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[5]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[6]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[7]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[8]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[9]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[10]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[11]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[12]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[13]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[14]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[15]);



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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[16]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[17]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[18]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[19]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[20]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[21]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[22]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[23]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[24]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[25]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[26]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[27]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[28]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[29]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[30]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[31]);


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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[32]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[33]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[34]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[35]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[36]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[37]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[38]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[39]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[40]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[41]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[42]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[43]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[44]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[45]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[46]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[47]);




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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[48]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[49]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[50]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[51]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[52]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[53]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[54]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[55]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[56]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[57]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[58]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[59]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[60]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[61]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[62]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[63]);

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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[0]);
	roundSha256(h, a, b, c, d, e, f, g, 0, _K[1]);
	roundSha256(g, h, a, b, c, d, e, f, 0, _K[2]);
	roundSha256(f, g, h, a, b, c, d, e, 0, _K[3]);
	roundSha256(e, f, g, h, a, b, c, d, 0, _K[4]);
	roundSha256(d, e, f, g, h, a, b, c, 0, _K[5]);
	roundSha256(c, d, e, f, g, h, a, b, 0, _K[6]);
	roundSha256(b, c, d, e, f, g, h, a, 0, _K[7]);
	roundSha256(a, b, c, d, e, f, g, h, 0, _K[8]);
	roundSha256(h, a, b, c, d, e, f, g, 0, _K[9]);
	roundSha256(g, h, a, b, c, d, e, f, 0, _K[10]);
	roundSha256(f, g, h, a, b, c, d, e, 0, _K[11]);
	roundSha256(e, f, g, h, a, b, c, d, 0, _K[12]);
	roundSha256(d, e, f, g, h, a, b, c, 0, _K[13]);
	roundSha256(c, d, e, f, g, h, a, b, 0, _K[14]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[15]);

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
	


	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[16]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[17]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[18]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[19]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[20]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[21]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[22]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[23]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[24]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[25]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[26]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[27]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[28]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[29]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[30]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[31]);

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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[32]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[33]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[34]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[35]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[36]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[37]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[38]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[39]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[40]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[41]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[42]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[43]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[44]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[45]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[46]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[47]);

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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[48]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[49]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[50]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[51]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[52]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[53]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[54]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[55]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[56]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[57]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[58]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[59]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[60]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[61]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[62]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[63]);

	digest[0] = tmp[0] + a;
	digest[1] = tmp[1] + b;
	digest[2] = tmp[2] + c;
	digest[3] = tmp[3] + d;
	digest[4] = tmp[4] + e;
	digest[5] = tmp[5] + f;
	digest[6] = tmp[6] + g;
	digest[7] = tmp[7] + h;
}

__device__ void sha256PublicKeyCompressed(const unsigned int x[8], unsigned int yParity, unsigned int digest[8])
{
	unsigned int a, b, c, d, e, f, g, h;
	unsigned int w[16];

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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[0]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[1]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[2]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[3]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[4]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[5]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[6]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[7]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[8]);
	roundSha256(h, a, b, c, d, e, f, g, 0, _K[9]);
	roundSha256(g, h, a, b, c, d, e, f, 0, _K[10]);
	roundSha256(f, g, h, a, b, c, d, e, 0, _K[11]);
	roundSha256(e, f, g, h, a, b, c, d, 0, _K[12]);
	roundSha256(d, e, f, g, h, a, b, c, 0, _K[13]);
	roundSha256(c, d, e, f, g, h, a, b, 0, _K[14]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[15]);

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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[16]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[17]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[18]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[19]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[20]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[21]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[22]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[23]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[24]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[25]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[26]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[27]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[28]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[29]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[30]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[31]);

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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[32]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[33]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[34]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[35]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[36]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[37]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[38]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[39]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[40]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[41]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[42]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[43]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[44]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[45]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[46]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[47]);


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

	roundSha256(a, b, c, d, e, f, g, h, w[0], _K[48]);
	roundSha256(h, a, b, c, d, e, f, g, w[1], _K[49]);
	roundSha256(g, h, a, b, c, d, e, f, w[2], _K[50]);
	roundSha256(f, g, h, a, b, c, d, e, w[3], _K[51]);
	roundSha256(e, f, g, h, a, b, c, d, w[4], _K[52]);
	roundSha256(d, e, f, g, h, a, b, c, w[5], _K[53]);
	roundSha256(c, d, e, f, g, h, a, b, w[6], _K[54]);
	roundSha256(b, c, d, e, f, g, h, a, w[7], _K[55]);
	roundSha256(a, b, c, d, e, f, g, h, w[8], _K[56]);
	roundSha256(h, a, b, c, d, e, f, g, w[9], _K[57]);
	roundSha256(g, h, a, b, c, d, e, f, w[10], _K[58]);
	roundSha256(f, g, h, a, b, c, d, e, w[11], _K[59]);
	roundSha256(e, f, g, h, a, b, c, d, w[12], _K[60]);
	roundSha256(d, e, f, g, h, a, b, c, w[13], _K[61]);
	roundSha256(c, d, e, f, g, h, a, b, w[14], _K[62]);
	roundSha256(b, c, d, e, f, g, h, a, w[15], _K[63]);

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
