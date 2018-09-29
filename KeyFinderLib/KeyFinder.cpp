#include <fstream>
#include <iostream>

#include "KeyFinder.h"
#include "util.h"
#include "AddressUtil.h"

#include "Logger.h"


void KeyFinder::defaultResultCallback(KeySearchResult result)
{
	// Do nothing
}

void KeyFinder::defaultStatusCallback(KeySearchStatus status)
{
	// Do nothing
}

KeyFinder::KeyFinder(const secp256k1::uint256 &start, uint64_t range, int compression, KeySearchDevice* device)
{
	_total = 0;
	_statusInterval = 1000;
	_device = device;

	_compression = compression;

	_startExponent = start;

	_range = range;

	_statusCallback = NULL;

	_resultCallback = NULL;

    _iterCount = 0;
}

KeyFinder::~KeyFinder()
{
    delete _device;
}

void KeyFinder::setTargets(std::vector<std::string> &targets)
{
	if(targets.size() == 0) {
		throw KeySearchException("Requires at least 1 target");
	}

	_targets.clear();

	// Convert each address from base58 encoded form to a 160-bit integer
	for(unsigned int i = 0; i < targets.size(); i++) {

		if(!Address::verifyAddress(targets[i])) {
			throw KeySearchException("Invalid address '" + targets[i] + "'");
		}

		KeySearchTarget t;

		Base58::toHash160(targets[i], t.value);

		_targets.insert(t);
	}

    _device->setTargets(_targets);
}

void KeyFinder::setTargets(std::string targetsFile)
{
	std::ifstream inFile(targetsFile.c_str());

	if(!inFile.is_open()) {
		Logger::log(LogLevel::Error, "Unable to open '" + targetsFile + "'");
		throw KeySearchException();
	}

	_targets.clear();

	std::string line;
	Logger::log(LogLevel::Info, "Loading addresses from '" + targetsFile + "'");
	while(std::getline(inFile, line)) {
		util::removeNewline(line);
		if(line.length() > 0) {
			if(!Address::verifyAddress(line)) {
				Logger::log(LogLevel::Error, "Invalid address '" + line + "'");
				throw KeySearchException();
			}

			KeySearchTarget t;

			Base58::toHash160(line, t.value);

			_targets.insert(t);
		}
	}
	Logger::log(LogLevel::Info, util::formatThousands(_targets.size()) + " addresses loaded ("
		+ util::format("%.1f", (double)(sizeof(KeySearchTarget) * _targets.size()) / (double)(1024 * 1024)) + "MB)");

    _device->setTargets(_targets);
}


void KeyFinder::setResultCallback(void(*callback)(KeySearchResult))
{
	_resultCallback = callback;
}

void KeyFinder::setStatusCallback(void(*callback)(KeySearchStatus))
{
	_statusCallback = callback;
}

void KeyFinder::setStatusInterval(unsigned int interval)
{
	_statusInterval = interval;
}

void KeyFinder::setTargetsOnDevice()
{
	// Set the target in constant memory
	std::vector<struct hash160> targets;

	for(std::set<KeySearchTarget>::iterator i = _targets.begin(); i != _targets.end(); ++i) {
		targets.push_back(hash160((*i).value));
	}

    _device->setTargets(_targets);
}

void KeyFinder::init()
{
	Logger::log(LogLevel::Info, "Initializing " + _device->getDeviceName());

    _device->init(_startExponent, _compression);
}


void KeyFinder::stop()
{
	_running = false;
}

void KeyFinder::removeTargetFromList(const unsigned int hash[5])
{
	KeySearchTarget t(hash);

	_targets.erase(t);
}

bool KeyFinder::isTargetInList(const unsigned int hash[5])
{
	KeySearchTarget t(hash);
	return _targets.find(t) != _targets.end();
}


void KeyFinder::run()
{
    uint64_t pointsPerIteration = _device->keysPerIteration();

	_running = true;

	util::Timer timer;

	timer.start();

	uint64_t prevIterCount = 0;

	_totalTime = 0;

	while(_running) {

        _device->doStep();
        _iterCount++;

		// Update status
		unsigned int t = timer.getTime();

		if(t >= _statusInterval) {

			KeySearchStatus info;

			uint64_t count = (_iterCount - prevIterCount) * pointsPerIteration;

			_total += count;

			double seconds = (double)t / 1000.0;

			info.speed = (double)((double)count / seconds) / 1000000.0;

			info.total = _total;

			info.totalTime = _totalTime;

			size_t freeMem = 0;

			size_t totalMem = 0;

			_device->getMemoryInfo(freeMem, totalMem);

			info.freeMemory = freeMem;
			info.deviceMemory = totalMem;
			info.deviceName = _device->getDeviceName();
			info.targets = _targets.size();

			_statusCallback(info);

			timer.start();
			prevIterCount = _iterCount;
			_totalTime += t;
		}

        std::vector<KeySearchResult> results;

        if(_device->getResults(results) > 0) {

			for(unsigned int i = 0; i < results.size(); i++) {

				KeySearchResult info;
                info.privateKey = results[i].privateKey;
                info.publicKey = results[i].publicKey;
				info.compressed = results[i].compressed;
				info.address = Address::fromPublicKey(results[i].publicKey, results[i].compressed);

				_resultCallback(info);
			}

			// Remove the hashes that were found
			for(unsigned int i = 0; i < results.size(); i++) {
				removeTargetFromList(results[i].hash);
			}
		}

		// Stop if we searched the entire range, or have no targets left
		if((_range > 0 && _iterCount * pointsPerIteration >= _range) || _targets.size() == 0) {
			_running = false;
		}
	}
}