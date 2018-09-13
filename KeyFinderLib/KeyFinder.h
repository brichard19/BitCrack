#ifndef _KEY_FINDER_H
#define _KEY_FINDER_H

#include <stdint.h>
#include <vector>
#include <set>
#include "secp256k1.h"
#include "atomiclist.h"
#include "hashlookup.h"
#include "ec.h"

class CudaDeviceContext;

struct KeyFinderResult {
	int thread;
	int block;
	int index;
	bool compressed;

	secp256k1::ecpoint p;
	unsigned int hash[5];
};

typedef struct {
	std::string address;
	secp256k1::ecpoint publicKey;
	secp256k1::uint256 privateKey;
	bool compressed;
}KeyFinderResultInfo;



typedef struct {
	int device;
	double speed;
	uint64_t total;
	uint64_t totalTime;
	std::string deviceName;
	uint64_t freeMemory;
	uint64_t deviceMemory;
	size_t targets;
}KeyFinderStatusInfo;


class KeyFinderTarget {

public:
	unsigned int value[5];

	KeyFinderTarget()
	{
		memset(value, 0, sizeof(value));
	}

	KeyFinderTarget(const unsigned int h[5])
	{
		for(int i = 0; i < 5; i++) {
			value[i] = h[i];
		}
	}


	bool operator==(const KeyFinderTarget &t) const
	{
		for(int i = 0; i < 5; i++) {
			if(value[i] != t.value[i]) {
				return false;
			}
		}

		return true;
	}

	bool operator<(const KeyFinderTarget &t) const
	{
		for(int i = 0; i < 5; i++) {
			if(value[i] < t.value[i]) {
				return true;
			} else if(value[i] > t.value[i]) {
				return false;
			}
		}

		return false;
	}

	bool operator>(const KeyFinderTarget &t) const
	{
		for(int i = 0; i < 5; i++) {
			if(value[i] > t.value[i]) {
				return true;
			} else if(value[i] < t.value[i]) {
				return false;
			}
		}

		return false;
	}
};

class KeyFinderException {

public:

	KeyFinderException()
	{

	}

	KeyFinderException(const std::string &msg)
	{
		this->msg = msg;
	}

	std::string msg;
};

class KeyFinder {

private:

	CudaDeviceKeys _deviceKeys;

	CudaAtomicList _resultList;

	CudaHashLookup _targetLookup;

	unsigned int _compression;

	unsigned int _flags;

	std::set<KeyFinderTarget> _targets;

	unsigned int _statusInterval;

	CudaDeviceContext *_devCtx;

	uint64_t _iterCount;
	uint64_t _total;
	unsigned int _totalTime;


	// CUDA blocks and threads
	int _numThreads;
	int _numBlocks;
	int _pointsPerThread;
	int _device;

	secp256k1::uint256 _startExponent;
	uint64_t _range;

	// Exponent/point pairs
	std::vector<secp256k1::uint256> _exponents;
	std::vector<secp256k1::ecpoint> _startingPoints;


	// Each index of each thread gets a flag to indicate if it found a valid hash
	unsigned int *_hashFoundFlags;

	bool _running;

	void(*_resultCallback)(KeyFinderResultInfo);
	void(*_statusCallback)(KeyFinderStatusInfo);


	static void defaultResultCallback(KeyFinderResultInfo result);
	static void defaultStatusCallback(KeyFinderStatusInfo status);


	void generateStartingPoints();

	bool verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5], bool compressed);

	void getResults(std::vector<KeyFinderResult> &r);

	void removeTargetFromList(const unsigned int value[5]);
	bool isTargetInList(const unsigned int value[5]);
	void setTargetsOnDevice();

    void cudaCall(cudaError_t err);

public:
	class Compression {
	public:
		enum {
			COMPRESSED = 0,
			UNCOMPRESSED = 1,
			BOTH = 2,
		};
	};

	KeyFinder(int device, const secp256k1::uint256 &start, uint64_t range, int compression, int blocks = 0, int threads = 0, int pointsPerThread = 0);

	~KeyFinder();

	void init();
	void run();
	void stop();

	void setResultCallback(void(*callback)(KeyFinderResultInfo));
	void setStatusCallback(void(*callback)(KeyFinderStatusInfo));
	void setStatusInterval(unsigned int interval);

	void setTargets(std::string targetFile);
	void setTargets(std::vector<std::string> &targets);

};

#endif