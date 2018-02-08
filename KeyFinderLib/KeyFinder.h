#ifndef _KEY_FINDER_H
#define _KEY_FINDER_H

#include <vector>
#include "secp256k1.h"


class DeviceContext;

typedef struct {
	secp256k1::ecpoint publicKey;
	secp256k1::uint256 privateKey;

}KeyFinderResultInfo;



typedef struct {
	int device;
	double speed;
	unsigned long long total;
	unsigned int totalTime;
}KeyFinderStatusInfo;


struct KeyFinderTarget {
	secp256k1::ecpoint p;
	unsigned int hash[5];
};

class KeyFinderException {

public:

	KeyFinderException(const std::string &msg)
	{
		this->msg = msg;
	}

	std::string msg;
};


class KeyFinder {

private:

	const int MODE_ADDRESS = 0;
	const int MODE_PUBKEY = 1;

	std::vector<KeyFinderTarget> _targets;

	unsigned int _statusInterval;

	DeviceContext *_devCtx;

	unsigned long long _iterCount;
	unsigned long long _total;
	unsigned int _totalTime;


	// CUDA blocks and threads
	int _numThreads;
	int _numBlocks;
	int _pointsPerThread;
	int _device;

	secp256k1::uint256 _startExponent;
	unsigned long long _range;

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

	bool verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5]);


public:

	KeyFinder(const secp256k1::uint256 &start, unsigned long long range, std::vector<std::string> &targetHashes, int blocks = 0, int threads = 0, int pointsPerThread = 0);
	~KeyFinder();

	void init();
	void run();
	void stop();

	void setResultCallback(void(*callback)(KeyFinderResultInfo));
	void setStatusCallback(void(*callback)(KeyFinderStatusInfo));
	void setStatusInterval(unsigned int interval);
};

#endif