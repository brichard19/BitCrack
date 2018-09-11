#include <fstream>
#include <iostream>

#include "CudaDeviceContext.h"

#include "KeyFinder.h"
#include "util.h"
#include "AddressUtil.h"

#include "KeyFinderShared.h"

#include "cudabridge.h"

#include "hashlookup.h"

#include "atomiclist.h"

#include "Logger.h"

#include "ec.h"
#include "cudaUtil.h"

void KeyFinder::defaultResultCallback(KeyFinderResultInfo result)
{
	// Do nothing
}

void KeyFinder::defaultStatusCallback(KeyFinderStatusInfo status)
{
	// Do nothing
}

KeyFinder::KeyFinder(int device, const secp256k1::uint256 &start, unsigned long long range, int compression, int blocks, int threads, int pointsPerThread)
{
	_devCtx = NULL;
	_total = 0;
	_statusInterval = 1000;
	_device = device;


	if(threads <= 0 || threads % 32 != 0) {
		throw KeyFinderException("The number of threads must be a multiple of 32");
	}

	if(blocks <= 0) {
		throw KeyFinderException("At least 1 block required");
	}

	if(pointsPerThread <= 0) {
		throw KeyFinderException("At least 1 point per thread required");
	}

	if(!(compression == Compression::COMPRESSED || compression == Compression::UNCOMPRESSED || compression == Compression::BOTH)) {
		throw KeyFinderException("Invalid argument for compression");
	}

	if(start.cmp(secp256k1::N) >= 0) {
		throw KeyFinderException("Starting key is out of range");
	}


	_compression = compression;

	_numThreads = threads;

	_numBlocks = blocks;

	_pointsPerThread = pointsPerThread;

	_startExponent = start;

	_range = range;

	_statusCallback = NULL;

	_resultCallback = NULL;
}

KeyFinder::~KeyFinder()
{
	cleanupChainBuf();

	if(_devCtx) {
		delete _devCtx;
	}
}

void KeyFinder::setTargets(std::vector<std::string> &targets)
{
	if(targets.size() == 0) {
		throw KeyFinderException("Requires at least 1 target");
	}

	_targets.clear();

	// Convert each address from base58 encoded form to a 160-bit integer
	for(unsigned int i = 0; i < targets.size(); i++) {

		if(!Address::verifyAddress(targets[i])) {
			throw KeyFinderException("Invalid address '" + targets[i] + "'");
		}

		KeyFinderTarget t;

		Base58::toHash160(targets[i], t.value);

		_targets.insert(t);
	}
}

void KeyFinder::setTargets(std::string targetsFile)
{
	std::ifstream inFile(targetsFile.c_str());

	if(!inFile.is_open()) {
		Logger::log(LogLevel::Error, "Unable to open '" + targetsFile + "'");
		throw KeyFinderException();
	}

	_targets.clear();

	std::string line;
	Logger::log(LogLevel::Info, "Loading addresses from '" + targetsFile + "'");
	while(std::getline(inFile, line)) {
		if(line.length() > 0) {
			if(!Address::verifyAddress(line)) {
				Logger::log(LogLevel::Error, "Invalid address '" + line + "'");
				throw KeyFinderException();
			}

			KeyFinderTarget t;

			Base58::toHash160(line, t.value);

			_targets.insert(t);
		}
	}
	Logger::log(LogLevel::Info, util::formatThousands(_targets.size()) + " addresses loaded ("
		+ util::format("%.1f", (double)(sizeof(KeyFinderTarget) * _targets.size()) / (double)(1024 * 1024)) + "MB)");
}


void KeyFinder::setResultCallback(void(*callback)(KeyFinderResultInfo))
{
	_resultCallback = callback;
}

void KeyFinder::setStatusCallback(void(*callback)(KeyFinderStatusInfo))
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

	for(std::set<KeyFinderTarget>::iterator i = _targets.begin(); i != _targets.end(); ++i) {
		targets.push_back(hash160((*i).value));
	}

	cudaError_t err = _targetLookup.setTargets(targets);

	if(err) {
        std::string errStr = cudaGetErrorString(err);

        throw KeyFinderException(errStr);
	}
}

void KeyFinder::init()
{
	DeviceParameters params;
	params.device = _device;
	params.threads = _numThreads;
	params.blocks = _numBlocks;
	params.pointsPerThread = _pointsPerThread;

	// Allocate device memory
	_devCtx = new CudaDeviceContext(params);

	Logger::log(LogLevel::Info, "Initializing " + _devCtx->getDeviceName());

	_devCtx->init();

    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// Copy points to device
	generateStartingPoints();

	setTargetsOnDevice();

	allocateChainBuf(_numThreads * _numBlocks * _pointsPerThread);

	// Set the incrementor
	secp256k1::ecpoint g = secp256k1::G();
	secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256(_numThreads * _numBlocks * _pointsPerThread), g);

	cudaError_t err = _resultList.init(sizeof(KeyFinderDeviceResult), 16);

	if(err) {
		std::string cudaErrorString(cudaGetErrorString(err));

		throw KeyFinderException("Error initializing device: " + cudaErrorString);
	}

	err = setIncrementorPoint(p.x, p.y);

	if(err) {
		std::string cudaErrorString(cudaGetErrorString(err));

		throw KeyFinderException("Error initializing device: " + cudaErrorString);
	}
}


void KeyFinder::generateStartingPoints()
{
	_exponents.clear();
	_startingPoints.clear();
	_iterCount = 0;

	unsigned long long totalPoints = _pointsPerThread * _numThreads * _numBlocks;
	unsigned long long totalMemory = totalPoints * 40;

	Logger::log(LogLevel::Info, "Generating " + util::formatThousands(totalPoints) + " starting points (" + util::format("%.1f", (double)totalMemory / (double)(1024 * 1024)) + "MB)");

	// Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
	secp256k1::uint256 privKey = _startExponent;

	for(unsigned long long i = 0; i < totalPoints; i++) {
		_exponents.push_back(privKey.add(i));
	}

	_deviceKeys.init(_numBlocks, _numThreads, _pointsPerThread, _exponents);

	// Show progress in 10% increments
	double pct = 10.0;
	for(int i = 1; i <= 256; i++) {
		_deviceKeys.doStep();
		if(((double)i / 256.0) * 100.0 >= pct) {
			Logger::log(LogLevel::Info, util::format("%.1f%%", pct));
			pct += 10.0;
		}
	}

#ifdef _DEBUG
	try {
        Logger::log(LogLevel::Debug, "Verifying points on device. This will take a while...");
		_deviceKeys.selfTest(_exponents);
	} catch(std::string &e) {
		Logger::log(LogLevel::Debug, e);
		exit(1);
	}
#endif

	Logger::log(LogLevel::Info, "Done");

	_deviceKeys.clearPrivateKeys();
}


void KeyFinder::stop()
{
	_running = false;
}

/**
 Verified this private key produces this public key and hash
 */
bool KeyFinder::verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5], bool compressed)
{
	secp256k1::ecpoint g = secp256k1::G();

	secp256k1::ecpoint p = secp256k1::multiplyPoint(privateKey, g);

	if(!(p == publicKey)) {
		return false;
	}

	unsigned int xWords[8];
	unsigned int yWords[8];

	p.x.exportWords(xWords, 8, secp256k1::uint256::BigEndian);
	p.y.exportWords(yWords, 8, secp256k1::uint256::BigEndian);

	unsigned int digest[5];
	if(compressed) {
		Hash::hashPublicKeyCompressed(xWords, yWords, digest);
	} else {
		Hash::hashPublicKey(xWords, yWords, digest);
	}
	

	for(int i = 0; i < 5; i++) {
		if(digest[i] != hash[i]) {
			return false;
		}
	}
	
	return true;
}

void KeyFinder::removeTargetFromList(const unsigned int hash[5])
{
	KeyFinderTarget t(hash);

	_targets.erase(t);
}

bool KeyFinder::isTargetInList(const unsigned int hash[5])
{
	KeyFinderTarget t(hash);
	return _targets.find(t) != _targets.end();
}

void KeyFinder::getResults(std::vector<KeyFinderResult> &r)
{
	int count = _resultList.size();

	if(count == 0) {
		return;
	}

	unsigned char *ptr = new unsigned char[count * sizeof(KeyFinderDeviceResult)];

	_resultList.read(ptr, count);

	for(int i = 0; i < count; i++) {
		struct KeyFinderDeviceResult *rPtr = &((struct KeyFinderDeviceResult *)ptr)[i];

		// might be false-positive
		if(!isTargetInList(rPtr->digest)) {
			continue;
		}

		KeyFinderResult minerResult;
		minerResult.block = rPtr->block;
		minerResult.thread = rPtr->thread;
		minerResult.index = rPtr->idx;
		minerResult.compressed = rPtr->compressed;
		for(int j = 0; j < 5; j++) {
			minerResult.hash[j] = rPtr->digest[j];
		}
		minerResult.p = secp256k1::ecpoint(secp256k1::uint256(rPtr->x, secp256k1::uint256::BigEndian), secp256k1::uint256(rPtr->y, secp256k1::uint256::BigEndian));

		r.push_back(minerResult);
	}

	delete[] ptr;

	_resultList.clear();
}

void KeyFinder::run()
{
	unsigned int pointsPerIteration = _numBlocks * _numThreads * _pointsPerThread;

	_running = true;

	util::Timer timer;

	timer.start();

	unsigned long long prevIterCount = 0;

	_totalTime = 0;

	while(_running) {

		_resultList.clear();

		KernelParams params = _devCtx->getKernelParams();

        try {
            if(_iterCount < 2 && _startExponent.cmp(pointsPerIteration) <= 0) {
                callKeyFinderKernel(params, true, _compression);
            } else {
                callKeyFinderKernel(params, false, _compression);
            }
        } catch(cuda::CudaException &ex) {
            throw KeyFinderException(ex.msg);
        }
        
		// Update status
		unsigned int t = timer.getTime();

		if(t >= _statusInterval) {

			KeyFinderStatusInfo info;

			unsigned long long count = (_iterCount - prevIterCount) * pointsPerIteration;

			_total += count;

			double seconds = (double)t / 1000.0;

			info.speed = (double)((double)count / seconds) / 1000000.0;

			info.total = _total;

			info.totalTime = _totalTime;

			size_t freeMem = 0;

			size_t totalMem = 0;

			try {
				_devCtx->getMemInfo(freeMem, totalMem);
			} catch(DeviceContextException ex) {
				Logger::log(LogLevel::Error, "Error querying device memory: " + ex.msg);
			}

			info.freeMemory = freeMem;
			info.deviceMemory = totalMem;
			info.deviceName = _devCtx->getDeviceName();
			info.targets = _targets.size();

			_statusCallback(info);

			timer.start();
			prevIterCount = _iterCount;
			_totalTime += t;
		}


		// Report any results
		if(_resultList.size() > 0) {
			std::vector<KeyFinderResult> results;

			getResults(results);

			if(results.size() > 0) {

				for(unsigned int i = 0; i < results.size(); i++) {
					unsigned int index = _devCtx->getIndex(results[i].block, results[i].thread, results[i].index);

					secp256k1::uint256 exp = _exponents[index];
					secp256k1::ecpoint publicKey = results[i].p;

					unsigned long long offset = (unsigned long long)_numBlocks * _numThreads * _pointsPerThread * _iterCount;
					exp = secp256k1::addModN(exp, secp256k1::uint256(offset));

					if(!verifyKey(exp, publicKey, results[i].hash, results[i].compressed)) {
						throw KeyFinderException("Invalid point");
					}

					KeyFinderResultInfo info;
					info.privateKey = exp;
					info.publicKey = publicKey;
					info.compressed = results[i].compressed;
					info.address = Address::fromPublicKey(publicKey, results[i].compressed);

					_resultCallback(info);
				}

				// Remove the hashes that were found
				for(unsigned int i = 0; i < results.size(); i++) {
					removeTargetFromList(results[i].hash);
				}

				// Update hash targets on device
				if(_targets.size() > 0) {
					Logger::log(LogLevel::Info, "Reloading targets");
					setTargetsOnDevice();
				}
			}
		}
		_iterCount++;

		// Stop if we searched the entire range, or have no targets left
		if((_range > 0 && _iterCount * pointsPerIteration >= _range) || _targets.size() == 0) {
			_running = false;
		}
	}
}