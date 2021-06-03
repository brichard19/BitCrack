#ifndef KEY_FINDER_SHARED_H
#define KEY_FINDER_SHARED_H

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

#endif
