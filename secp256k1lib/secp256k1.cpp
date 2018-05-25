#include<string.h>
#include<stdio.h>
#include<stdlib.h>
#include"CryptoUtil.h"

#include "secp256k1.h"


using namespace secp256k1;

static uint256 _ONE(1);
static uint256 _ZERO;
static crypto::Rng _rng;

static inline void addc(unsigned int a, unsigned int b, unsigned int carryIn, unsigned int &sum, int &carryOut)
{
	unsigned long long sum64 = (unsigned long long)a + b + carryIn;

	sum = (unsigned int)sum64;
	carryOut = (int)(sum64 >> 32) & 1;
}


static inline void subc(unsigned int a, unsigned int b, unsigned int borrowIn, unsigned int &diff, int &borrowOut)
{
	unsigned long long diff64 = (unsigned long long)a - b - borrowIn;

	diff = (unsigned int)diff64;
	borrowOut = (int)((diff64 >> 32) & 1);
}



static bool lessThanEqualTo(const unsigned int *a, const unsigned int *b, int len)
{
	for(int i = len - 1; i >= 0; i--) {
		if(a[i] < b[i]) {
			// is greater than
			return true;
		} else if(a[i] > b[i]) {
			// is less than
			return false;
		}
	}

	// is equal
	return true;
}

static bool greaterThanEqualTo(const unsigned int *a, const unsigned int *b, int len)
{
	for(int i = len - 1; i >= 0; i--) {
		if(a[i] > b[i]) {
			// is greater than
			return true;
		} else if(a[i] < b[i]) {
			// is less than
			return false;
		}
	}

	// is equal
	return true;
}

static int add(const unsigned int *a, const unsigned int *b, unsigned int *c, int len)
{
	int carry = 0;

	for(int i = 0; i < len; i++) {
		addc(a[i], b[i], carry, c[i], carry);
	}

	return carry;
}

static int sub(const unsigned int *a, const unsigned int *b, unsigned int *c, int len)
{
	int borrow = 0;

	for(int i = 0; i < len; i++) {
		subc(a[i], b[i], borrow, c[i], borrow);
	}

	return borrow & 1;
}

static void multiply(const unsigned int *x, int xLen, const unsigned int *y, int yLen, unsigned int *z)
{
	// Zero out the first yLen words of the z. We only need to clear the first yLen words
	// because after each iteration the most significant word is overwritten anyway
	//memset(z, 0, (yLen + xLen) * sizeof(unsigned int));

	for(int i = 0; i < xLen + yLen; i++) {
		z[i] = 0;
	}

	int i, j;
	for(i = 0; i < xLen; i++) {

		unsigned int high = 0;

		for(j = 0; j < yLen; j++) {

			unsigned long long product = (unsigned long long)x[i] * y[j];

			// Take the existing sum and add it to the product plus the high word from the
			// previous multiplication. Since we are adding to a larger datatype, the compiler
			// will take care of performing any carries resulting from the addition
			product = product + z[i + j] + high;
			// update the sum
			z[i + j] = (unsigned int)product;

			// Keep the high word for the next iteration
			high = product >> 32;
		}

		z[i + yLen] = high;
	}
}

static uint256 rightShift(const uint256 &x, int count)
{
	uint256 r;

	count &= 0x1f;

	for(int i = 0; i < 7; i++) {
		r.v[i] = (x.v[i] >> count) | (x.v[i + 1] << (32 - count));
	}
	r.v[7] = x.v[7] >> count;

	return r;
}

uint256 uint256::mul(const uint256 &x) const
{
	unsigned int product[16] = { 0 };

	multiply(this->v, 8, x.v, 8, product);

	return uint256(product);
}

uint256 uint256::mul(int i) const
{
	unsigned int product[16] = { 0 };

	multiply((unsigned int *)&i, 1, this->v, 8, product);

	return uint256(product);
}

uint256 uint256::div(int val) const
{
	//uint256 sum;

	//unsigned int mask = 0x80000000;

	//for(int i = 31; i >= 0; i--) {
	//	if(val & mask) {
	//		uint256 k = *this;
	//		uint256 shifted = rightShift(k, i);
	//		sum = sum.add(shifted);
	//	}
	//	mask >>= 1;
	//}

	//return sum;

	uint256 t = *this;
	uint256 quotient;

	// Shift divisor left until MSB is 1
	unsigned int kWords[8] = { 0 };
	kWords[7] = val;

	int shiftCount = 7 * 32;

	while((kWords[7] & 0x80000000) == 0) {
		kWords[7] <<= 1;
		shiftCount++;
	}

	uint256 k(kWords);
	// while t >= divisor
	while(t.cmp(uint256(val)) >= 0) {

		// while t < k
		while(t.cmp(k) < 0) {
			// k = k / 2
			k = rightShift(k, 1);
			shiftCount--;
		}
		// t = t - k
		::sub(t.v, k.v, t.v, 8);

		quotient = quotient.add(uint256(2).pow(shiftCount));
	}

	return quotient;
}

uint256 uint256::mod(int val) const
{
	uint256 quotient = this->div(val);

	uint256 product = quotient.mul(val);

	uint256 result;

	::sub(this->v, product.v, result.v, 8);

	return result;
}

uint256 uint256::add(int val) const
{
	uint256 result(val);

	::add(this->v, result.v, result.v, 8);

	return result;
}

uint256 uint256::add(unsigned int val) const
{
	uint256 result(val);

	::add(this->v, result.v, result.v, 8);

	return result;
}

uint256 uint256::add(unsigned long long val) const
{
	uint256 result(val);

	::add(this->v, result.v, result.v, 8);

	return result;
}

uint256 uint256::sub(int val) const
{
	uint256 result(val);

	::sub(this->v, result.v, result.v, 8);

	return result;
}

uint256 uint256::add(const uint256 &val) const
{
	uint256 result;

	::add(this->v, val.v, result.v, 8);

	return result;
}

static bool isOne(const uint256 &x)
{
	if(x.v[0] != 1) {
		return false;
	}

	for(int i = 1; i < 8; i++) {
		if(x.v[i] != 0) {
			return false;
		}
	}

	return true;
}

static uint256 divBy2(const uint256 &x)
{
	uint256 r;

	for(int i = 0; i < 7; i++) {
		r.v[i] = (x.v[i] >> 1) | (x.v[i + 1] << 31);
	}
	r.v[7] = x.v[7] >> 1;

	return r;
}


static bool isEven(const uint256 &x)
{
	return (x.v[0] & 1) == 0;
}

ecpoint secp256k1::pointAtInfinity()
{
	uint256 x(_POINT_AT_INFINITY_WORDS);

	return ecpoint(x, x);
}

ecpoint secp256k1::G()
{
	uint256 x(_GX_WORDS);
	uint256 y(_GY_WORDS);

	return ecpoint(x, y);
}

uint256 secp256k1::invModP(const uint256 &x)
{
	uint256 u = x;
	uint256 v = P;
	uint256 x1 = _ONE;
	uint256 x2 = _ZERO;

	// Signed part of the 256-bit words
	int x1Signed = 0;
	int x2Signed = 0;

	while(!isOne(u) && !isOne(v)) {

		while(isEven(u)) {

			u = divBy2(u);

			if(isEven(x1)) {
				x1 = divBy2(x1);

				// Shift right (signed bit is preserved)
				x1.v[7] |= ((unsigned int)x1Signed & 0x01) << 31;

				x1Signed >>= 1;
			} else {
				int carry = add(x1.v, P.v, x1.v, 8);

				x1 = divBy2(x1);

				x1Signed += carry;

				x1.v[7] |= ((unsigned int)x1Signed & 0x01) << 31;

				x1Signed >>= 1;
			}

		}

		while(isEven(v)) {

			v = divBy2(v);

			if(isEven(x2)) {

				x2 = divBy2(x2);

				x2.v[7] |= ((unsigned int)x2Signed & 0x01) << 31;

				x2Signed >>= 1;
			} else {
				int carry = add(x2.v, P.v, x2.v, 8);

				x2 = divBy2(x2);

				x2Signed += carry;

				x2.v[7] |= ((unsigned int)x2Signed & 0x01) << 31;

				x2Signed >>= 1;
			}
		}

		if(lessThanEqualTo(v.v, u.v, 8)) {
			sub(u.v, v.v, u.v, 8);

			// x1 = x1 - x2
			int borrow = sub(x1.v, x2.v, x1.v, 8);
			x1Signed -= x2Signed;
			x1Signed -= borrow;
		} else {
			sub(v.v, u.v, v.v, 8);
			int borrow = sub(x2.v, x1.v, x2.v, 8);
			x2Signed -= x1Signed;
			x2Signed -= borrow;
		}
	}

	uint256 output;

	if(isOne(u)) {
	
		while(x1Signed < 0) {
			x1Signed += add(x1.v, P.v, x1.v, 8);
		}
	
		while(x1Signed > 0) {
			x1Signed -= sub(x1.v, P.v, x1.v, 8);
		}
	
		for(int i = 0; i < 8; i++) {
			output.v[i] = x1.v[i];
		}
	
	} else {
	
		while(x2Signed < 0) {
			x2Signed += add(x2.v, P.v, x2.v,  8);
		}
	
		while(x2Signed > 0) {
			x2Signed -= sub(x2.v, P.v, x2.v, 8);
		}
	
		for(int i = 0; i < 8; i++) {
			output.v[i] = x2.v[i];
		}
	}

	return output;
}



uint256 secp256k1::addModP(const uint256 &a, const uint256 &b)
{
	uint256 sum;

	int overflow = add(a.v, b.v, sum.v, 8);

	// mod P
	if(overflow || greaterThanEqualTo(sum.v, P.v, 8)) {
		sub(sum.v, P.v, sum.v, 8);
	}

	return sum;
}

uint256 secp256k1::addModN(const uint256 &a, const uint256 &b)
{
	uint256 sum;

	int overflow = add(a.v, b.v, sum.v, 8);

	// mod P
	if(overflow || greaterThanEqualTo(sum.v, N.v, 8)) {
		sub(sum.v, N.v, sum.v, 8);
	}

	return sum;
}

uint256 secp256k1::subModN(const uint256 &a, const uint256 &b)
{
	uint256 diff;

	if(sub(a.v, b.v, diff.v, 8)) {
		add(diff.v, N.v, diff.v, 8);
	}

	return diff;
}

uint256 secp256k1::subModP(const uint256 &a, const uint256 &b)
{
	uint256 diff;

	if(sub(a.v, b.v, diff.v, 8)) {
		add(diff.v, P.v, diff.v, 8);
	}

	return diff;
}



uint256 secp256k1::negModP(const uint256 &x)
{
	return subModP(P, x);
}

uint256 secp256k1::negModN(const uint256 &x)
{
	return subModN(N, x);
}

uint256 secp256k1::multiplyModP(const uint256 &a, const uint256 &b)
{
	unsigned int product[16];

	multiply(a.v, 8, b.v, 8, product);

	unsigned int tmp[10] = { 0 };
	unsigned int tmp2[10] = { 0 };
	unsigned int s = 977;

	//multiply by high 8 words by 2^32 + 977
	for(int i = 0; i < 8; i++) {
		tmp2[1 + i] = product[8 + i];
	}

	multiply(&s, 1, &product[8], 8, &tmp[0]);
	add(tmp, tmp2, tmp, 10);

	// clear top 8 words of product
	for(int i = 8; i < 16; i++) {
		product[i] = 0;
	}

	//add to product
	add(&product[0], tmp, &product[0], 10);


	//multiply high 2 words by 2^32 + 977
	for(int i = 0; i < 8; i++) {
		tmp2[1 + i] = product[8 + i];
	}
	//multiply(&s, 1, &product[8], 2, &tmp[1]);
	multiply(&s, 1, &product[8], 8, &tmp[0]);
	add(tmp, tmp2, tmp, 10);



	// add to low 8 words
	int overflow = add(&product[0], &tmp[0], &product[0], 8);

	if(overflow || greaterThanEqualTo(&product[0], P.v, 8)) {
		sub(&product[0], P.v, &product[0], 8);
	}

	uint256 result;

	for(int i = 0; i < 8; i++) {
		result.v[i] = product[i];
	}

	return result;
}


static void reduceModN(const unsigned int *x, unsigned int *r)
{
	unsigned int barrettN[] = { 0x2fc9bec0, 0x402da173, 0x50b75fc4, 0x45512319, 0x00000001, 0x00000000, 0x00000000, 0x00000000, 00000001 };
	unsigned int product[25] = { 0 };

	// Multiply by barrett constant
	multiply(barrettN, 9, x, 16, product);

	// divide by 2^512
	for(int i = 0; i < 9; i++) {
		product[i] = product[16 + i];
	}

	unsigned int product2[16] = { 0 };

	// Multiply by N
	multiply(product, 8, N.v, 8, product2);

	// Take the difference
	unsigned int diff[16] = { 0 };
	sub(x, product2, diff, 16);

	if((diff[8] & 1) || greaterThanEqualTo(diff, N.v, 8)) {
		sub(diff, N.v, diff, 8);
	}

	for(int i = 0; i < 8; i++) {
		r[i] = diff[i];
	}
}

uint256 secp256k1::multiplyModN(const uint256 &a, const uint256 &b)
{
	unsigned int product[16];

	multiply(a.v, 8, b.v, 8, product);

	uint256 r;

	bool gt = false;
	for(int i = 0; i < 8; i++) {
		if(product[8 + i] != 0) {
			gt = true;
			break;
		}
	}

	if(gt) {
		reduceModN(product, r.v);
	} else if(greaterThanEqualTo(product, N.v, 8)) {
		sub(product, N.v, r.v, 8);
	} else {
		for(int i = 0; i < 8; i++) {
			r.v[i] = product[i];
		}
	}

	return r;
}

std::string secp256k1::uint256::toString(int base)
{
	std::string s = "";

	for(int i = 7; i >= 0; i--) {
		char hex[9] = { 0 };

		sprintf(hex, "%.8X", this->v[i]);
		s += std::string(hex);
	}

	return s;
}


uint256 secp256k1::generatePrivateKey()
{
	uint256 k;

	_rng.get((unsigned char *)k.v, 32);

	return k;
}

bool secp256k1::isPointAtInfinity(const ecpoint &p)
{

	for(int i = 0; i < 8; i++) {
		if(p.x.v[i] != 0xffffffff) {
			return false;
		}
	}

	for(int i = 0; i < 8; i++) {
		if(p.y.v[i] != 0xffffffff) {
			return false;
		}
	}

	return true;
}

ecpoint secp256k1::doublePoint(const ecpoint &p)
{
	// 1 / 2y
	uint256 yInv = invModP(addModP(p.y, p.y));

	// s = 3x^2 / 2y
	uint256 x3 = multiplyModP(p.x, p.x);
	uint256 s = multiplyModP(addModP(addModP(x3, x3), x3), yInv);

	//rx = s^2 - 2x
	uint256 rx = subModP(subModP(multiplyModP(s, s), p.x), p.x);

	//ry = s * (px - rx) - py
	uint256 ry = subModP(multiplyModP(s, subModP(p.x, rx)), p.y);

	ecpoint result;
	result.x = rx;
	result.y = ry;

	return result;
}

ecpoint secp256k1::addPoints(const ecpoint &p1, const ecpoint &p2)
{
	if(p1 == p2) {
		return doublePoint(p1);
	}

	if(p1.x == p2.x) {
		return pointAtInfinity();
	}

	if(isPointAtInfinity(p1)) {
		return p2;
	}

	if(isPointAtInfinity(p2)) {
		return p1;
	}

	uint256 rise = subModP(p1.y, p2.y);
	uint256 run = subModP(p1.x, p2.x);

	uint256 s = multiplyModP(rise, invModP(run));

	//rx = (s*s - px - qx) % _p;
	uint256 rx = subModP(subModP(multiplyModP(s, s), p1.x), p2.x);

	//ry = (s * (px - rx) - py) % _p;
	uint256 ry = subModP(multiplyModP(s, subModP(p1.x, rx)), p1.y);

	ecpoint sum;
	sum.x = rx;
	sum.y = ry;

	return sum;
}

ecpoint secp256k1::multiplyPoint(const uint256 &k, const ecpoint &p)
{
	ecpoint sum = pointAtInfinity();
	ecpoint d = p;

	for(int i = 0; i < 256; i++) {
		unsigned int mask = 1 << (i % 32);

		if(k.v[i / 32] & mask) {
			sum = addPoints(sum, d);
		}

		d = doublePoint(d);
	}

	return sum;
}

uint256 generatePrivateKey()
{
	uint256 k;

	for(int i = 0; i < 8; i++) {
		k.v[i] = ((unsigned int)rand() | ((unsigned int)rand()) << 17);
	}

	return k;
}

bool secp256k1::pointExists(const ecpoint &p)
{
	uint256 y = multiplyModP(p.y, p.y);

	uint256 x = addModP(multiplyModP(multiplyModP(p.x, p.x), p.x), uint256(7));

	return y == x;
}

static void bulkInversionModP(std::vector<uint256> &in)
{

	std::vector<uint256> products;
	uint256 total(1);

	for(unsigned int i = 0; i < in.size(); i++) {
		total = secp256k1::multiplyModP(total, in[i]);

		products.push_back(total);
	}

	// Do the inversion

	uint256 inverse = secp256k1::invModP(total);

	for(int i = (int)in.size() - 1; i >= 0; i--) {

		if(i > 0) {
			uint256 newValue = secp256k1::multiplyModP(products[i - 1], inverse);
			inverse = multiplyModP(inverse, in[i]);
			in[i] = newValue;
		} else {
			in[i] = inverse;
		}
	}
}

void secp256k1::generateKeypairsBulk(unsigned int count, const ecpoint &basePoint, std::vector<uint256> &privKeysOut, std::vector<ecpoint> &pubKeysOut)
{
	privKeysOut.clear();

	for(unsigned int i = 0; i < count; i++) {
		privKeysOut.push_back(generatePrivateKey());
	}

	generateKeyPairsBulk(basePoint, privKeysOut, pubKeysOut);
}

void secp256k1::generateKeyPairsBulk(const ecpoint &basePoint, std::vector<uint256> &privKeys, std::vector<ecpoint> &pubKeysOut)
{
	unsigned int count = (unsigned int)privKeys.size();

	//privKeysOut.clear();
	pubKeysOut.clear();

	// generate a table of points G, 2G, 4G, 8G...(2^255)G
	std::vector<ecpoint> table;

	table.push_back(basePoint);
	for(int i = 1; i < 256; i++) {

		ecpoint p = doublePoint(table[i-1]);
		if(!pointExists(p)) {
			throw "Point does not exist!";
		}
		table.push_back(p);
	}

	for(unsigned int i = 0; i < count; i++) {
		//privKeysOut.push_back(generatePrivateKey());
		pubKeysOut.push_back(ecpoint());
	}

	for(int i = 0; i < 256; i++) {

		std::vector<uint256> runList;

		// calculate (Px - Qx)
		for(unsigned int j = 0; j < count; j++) {
			uint256 run;
			uint256 k = privKeys[j];

			if(k.bit(i)) {
				if(isPointAtInfinity(pubKeysOut[j])) {
					run = uint256(2);
				} else {
					run = subModP(pubKeysOut[j].x, table[i].x);
				}
			} else {
				run = uint256(2);
			}

			runList.push_back(run);
		}

		// calculate 1/(Px - Qx)
		bulkInversionModP(runList);

		// complete the addition
		for(unsigned int j = 0; j < count; j++) {
			uint256 rise;
			uint256 k = privKeys[j];

			if(k.bit(i)) {
				if(isPointAtInfinity(pubKeysOut[j])) {
					pubKeysOut[j] = table[i];
				} else {
					rise = subModP(pubKeysOut[j].y, table[i].y);

					// s = (Py - Qy)/(Px - Qx)
					uint256 s = multiplyModP(rise, runList[j]);

					//rx = (s*s - px - qx) % _p;
					uint256 rx = subModP(subModP(multiplyModP(s, s), pubKeysOut[j].x), table[i].x);

					//ry = (s * (px - rx) - py) % _p;
					uint256 ry = subModP(multiplyModP(s, subModP(pubKeysOut[j].x, rx)), pubKeysOut[j].y);

					ecpoint r(rx, ry);
					if(!pointExists(r)) {
						throw "Point does not exist";
					}
					pubKeysOut[j] = r;
				}
			}
		}
	}
}

/**
 * Parses a public key. Expected format is 04<64 hex digits for X><64 hex digits for Y>
 */
secp256k1::ecpoint secp256k1::parsePublicKey(const std::string &pubKeyString)
{
	if(pubKeyString.length() != 130) {
		throw std::string("Invalid public key");
	}

	if(pubKeyString[0] != '0' || pubKeyString[1] != '4') {
		throw std::string("Invalid public key");
	}

	std::string xString = pubKeyString.substr(2, 64);
	std::string yString = pubKeyString.substr(66, 64);

	uint256 x(xString);
	uint256 y(yString);

	ecpoint p(x, y);

	if(!pointExists(p)) {
		throw std::string("Invalid public key");
	}

	return p;
}
