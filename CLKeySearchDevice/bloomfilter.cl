#ifndef BLOOMFILTER_CL
#define BLOOMFILTER_CL

bool isInBloomFilter(const unsigned int hash[5], __global unsigned int *targetList, const ulong *mask)
{
    unsigned int h5 = hash[0] + hash[1] + hash[2] + hash[3] + hash[4];

    return (false == 
        (
            (targetList[(((hash[0] << 6) | (h5 & 0x3f)) & *mask) / 32] & (0x01 << ((((hash[0] << 6) | (h5 & 0x3f)) & *mask) % 32))) == 0 ||
            (targetList[(((hash[1] << 6) | ((h5 >> 6) & 0x3f)) & *mask) / 32] & (0x01 << ((((hash[1] << 6) | ((h5 >> 6) & 0x3f)) & *mask) % 32))) == 0 ||
            (targetList[(((hash[2] << 6) | ((h5 >> 12) & 0x3f)) & *mask) / 32] & (0x01 << ((((hash[2] << 6) | ((h5 >> 12) & 0x3f)) & *mask) % 32))) == 0 ||
            (targetList[(((hash[3] << 6) | ((h5 >> 18) & 0x3f)) & *mask) / 32] & (0x01 << ((((hash[3] << 6) | ((h5 >> 18) & 0x3f)) & *mask) % 32))) == 0 || 
            (targetList[ (((hash[4] << 6) | ((h5 >> 24) & 0x3f)) & *mask) / 32] & (0x01 << ( (((hash[4] << 6) | ((h5 >> 24) & 0x3f)) & *mask) % 32))) == 0
        )
    );
}

#endif
