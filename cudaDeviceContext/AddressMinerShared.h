#ifndef _ADDRESS_MINER_SHARED_H
#define _ADDRESS_MINER_SHARED_H

namespace AutomorphismType {
	enum Value {
		NONE,
		NEGATIVE,
		TYPE1,
		TYPE1_NEGATIVE,
		TYPE2,
		TYPE2_NEGATIVE
	};
};

namespace PointCompressionType {
	enum Value {
		COMPRESSED = 1,
		UNCOMPRESSED = 2
	};
}


struct AddressMinerDeviceResult {
	int thread;
	int block;
	int idx;
	int automorphism;
	int compressed;
	unsigned int x[8];
	unsigned int y[8];
};


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