#ifndef endian
#define endian(x) ((x) << 24) | (((x) << 8) & 0x00ff0000) | (((x) >> 8) & 0x0000ff00) | ((x) >> 24)
#endif

void hashPublicKeyCompressed(uint256_t x, unsigned int yParity, unsigned int digest[5])
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

    ripemd160sha256NoFinal(hash, digest);
}

void hashPublicKey(uint256_t x, uint256_t y, unsigned int digest[5])
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

    ripemd160sha256NoFinal(hash, digest);
}
