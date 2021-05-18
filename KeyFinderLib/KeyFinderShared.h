#ifndef _KEY_FINDER_SHARED_H
#define _KEY_FINDER_SHARED_H

namespace PointCompressionType {
    enum Value {
        COMPRESSED = 0,
        UNCOMPRESSED = 1,
        BOTH = 2
    };
}

// Structures that exist on both host and device side
struct KeyFinderDeviceResult {
    int thread;
    int block;
    int idx;
    bool compressed;
    unsigned int x[8];
    unsigned int y[8];
    unsigned int digest[5];
};

//typedef struct hash160 {
//
//	unsigned int h[5];
//
//	hash160(const unsigned int hash[5])
//	{
//		memcpy(h, hash, sizeof(unsigned int) * 5);
//	}
//}hash160;

#endif