#ifndef _KEY_FINDER_H
#define _KEY_FINDER_H

#include <stdint.h>
#include <vector>
#include <set>
#include "secp256k1.h"
#include "KeySearchTypes.h"
#include "KeySearchDevice.h"


class KeyFinder {

private:

    KeySearchDevice *_device;

	unsigned int _compression;

	std::set<KeySearchTarget> _targets;

	unsigned int _statusInterval;

	uint64_t _iterCount;
	uint64_t _total;
	unsigned int _totalTime;

	secp256k1::uint256 _startExponent;
	uint64_t _range;

	// Each index of each thread gets a flag to indicate if it found a valid hash
	bool _running;

	void(*_resultCallback)(KeySearchResult);
	void(*_statusCallback)(KeySearchStatus);


	static void defaultResultCallback(KeySearchResult result);
	static void defaultStatusCallback(KeySearchStatus status);

	void removeTargetFromList(const unsigned int value[5]);
	bool isTargetInList(const unsigned int value[5]);
	void setTargetsOnDevice();

public:

    KeyFinder(const secp256k1::uint256 &start, uint64_t range, int compression, KeySearchDevice* device);

	~KeyFinder();

	void init();
	void run();
	void stop();

	void setResultCallback(void(*callback)(KeySearchResult));
	void setStatusCallback(void(*callback)(KeySearchStatus));
	void setStatusInterval(unsigned int interval);

	void setTargets(std::string targetFile);
	void setTargets(std::vector<std::string> &targets);
};

#endif