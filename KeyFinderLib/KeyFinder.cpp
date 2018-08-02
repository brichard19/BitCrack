#include <fstream>
#include <iostream>

#include "KeyFinder.h"
#include "util.h"
#include "AddressUtil.h"

#include "CudaDeviceContext.h"
#include "cudabridge.h"


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
		throw KeyFinderException("At least one block required");
	}

	if(pointsPerThread <= 0) {
		throw KeyFinderException("At least one point per thread required");
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
	cleanupTargets();
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
		throw KeyFinderException("Cannot open targets file");
	}

	_targets.clear();

	std::string line;

	while(std::getline(inFile, line)) {
		if(line.length() > 0) {
			if(!Address::verifyAddress(line)) {
				throw KeyFinderException("Invalid address '" + line + "'");
			}

			KeyFinderTarget t;

			Base58::toHash160(line, t.value);

			_targets.insert(t);
		}
	}
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

	cudaError_t err = setTargetHash(targets);
	if(err) {
		std::string cudaErrorString(cudaGetErrorString(err));

		throw KeyFinderException("Device error: " + cudaErrorString);
	}
}

void KeyFinder::init()
{
	// Allocate device memory
	_devCtx = new CudaDeviceContext;

	DeviceParameters params;
	params.device = _device;
	params.threads = _numThreads;
	params.blocks = _numBlocks;
	params.pointsPerThread = _pointsPerThread;

	_devCtx->init(params);

	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	// Copy points to device
	generateStartingPoints();
	_devCtx->copyPoints(_startingPoints);

	setTargetsOnDevice();

	allocateChainBuf(_numThreads * _numBlocks * _pointsPerThread);

	// Set the incrementor
	secp256k1::ecpoint g = secp256k1::G();
	secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256(_numThreads * _numBlocks * _pointsPerThread), g);


	cudaError_t err = setIncrementorPoint(p.x, p.y);
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

	// Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
	secp256k1::uint256 privKey = _startExponent;

	for(unsigned long long i = 0; i < totalPoints; i++) {
		_exponents.push_back(privKey.add(i));
	}

	secp256k1::generateKeyPairsBulk(secp256k1::G(), _exponents, _startingPoints);

	for(unsigned int i = 0; i < _startingPoints.size(); i++) {
		if(!secp256k1::pointExists(_startingPoints[i])) {
			throw KeyFinderException("Point generation error");
		}
	}
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
	int count = _devCtx->getResultCount();

	if(count == 0) {
		return;
	}

	unsigned char *ptr = new unsigned char[count * sizeof(KeyFinderDeviceResult)];

	_devCtx->getResults(ptr, count * sizeof(KeyFinderDeviceResult));


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
		for(int i = 0; i < 5; i++) {
			minerResult.hash[i] = rPtr->digest[i];
		}
		minerResult.p = secp256k1::ecpoint(secp256k1::uint256(rPtr->x, secp256k1::uint256::BigEndian), secp256k1::uint256(rPtr->y, secp256k1::uint256::BigEndian));

		r.push_back(minerResult);
	}

	delete[] ptr;

	_devCtx->clearResults();
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

		_devCtx->clearResults();

		KernelParams params = _devCtx->getKernelParams();
		if(_iterCount < 2 && _startExponent.cmp(pointsPerIteration) <= 0) {
			callKeyFinderKernel(params, true, _compression);
		} else {
			callKeyFinderKernel(params, false, _compression);
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

			if(cudaMemGetInfo(&freeMem, &totalMem)) {
				fprintf(stderr, "Error querying device memory usage\n");
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
		if(_devCtx->resultFound()) {
			std::vector<KeyFinderResult> results;

			getResults(results);


			if(results.size() > 0) 				{

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
				setTargetsOnDevice();
			}
		}
		_iterCount++;

		// Stop if we searched the entire range, or have no targets left
		if((_range > 0 && _iterCount * pointsPerIteration >= _range) || _targets.size() == 0) {
			_running = false;
		}
	}
}