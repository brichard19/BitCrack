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

#endif