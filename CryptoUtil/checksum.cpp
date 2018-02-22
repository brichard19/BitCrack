#include "CryptoUtil.h"
#include <stdio.h>
#include <string.h>

static unsigned int endian(unsigned int x)
{
	return (x << 24) | ((x << 8) & 0x00ff0000) | ((x >> 8) & 0x0000ff00) | (x >> 24);
}

unsigned int crypto::checksum(const unsigned int *hash)
{
	unsigned int msg[16] = { 0 };
	unsigned int digest[8] = { 0 };

	// Insert network byte, shift everything right 1 byte
	msg[0] = 0x00; // main network
	msg[0] |= hash[0] >> 8;
	msg[1] = (hash[0] << 24) | (hash[1] >> 8);
	msg[2] = (hash[1] << 24) | (hash[2] >> 8);
	msg[3] = (hash[2] << 24) | (hash[3] >> 8);
	msg[4] = (hash[3] << 24) | (hash[4] >> 8);
	msg[5] = (hash[4] << 24) | 0x00800000;

	// Padding and length
	msg[15] = 168;

	// Hash address
	sha256Init(digest);
	sha256(msg, digest);

	// Prepare to make a hash of the digest
	memset(msg, 0, 16 * sizeof(unsigned int));
	for(int i = 0; i < 8; i++) {
		msg[i] = digest[i];
	}

	msg[8] = 0x80000000;
	msg[15] = 256;


	sha256Init(digest);
	sha256(msg, digest);

	return digest[0];
}


