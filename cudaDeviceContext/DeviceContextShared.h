#ifndef _DEVICE_CONTEXT_SHARED_H
#define _DEVICE_CONTEXT_SHARED_H

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

struct hash160 {
	unsigned int h[5];
};

namespace PointCompressionType {
	enum Value {
		COMPRESSED = 0,
		UNCOMPRESSED = 1,
		BOTH = 2
	};
}

#endif