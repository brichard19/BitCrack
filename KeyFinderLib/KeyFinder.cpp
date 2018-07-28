#include "KeyFinder.h"
#include "util.h"
#include "AddressUtil.h"

#include "DeviceContext.h"
#include "cudabridge.h"

static const int DEFAULT_POINTS_PER_THREAD = 1;
static const int DEFAULT_NUM_THREADS = 32;
static const int DEFAULT_NUM_BLOCKS = 1;


void KeyFinder::defaultResultCallback(KeyFinderResultInfo result)
{
	// Do nothing
}

void KeyFinder::defaultStatusCallback(KeyFinderStatusInfo status)
{
	// Do nothing
}


KeyFinder::KeyFinder(int device, const secp256k1::uint256 &start, unsigned long long range, std::vector<std::string> &targetHashes, int compression, int blocks, int threads, int pointsPerThread)
{
	_devCtx = NULL;
	_total = 0;
	_statusInterval = 1000;
	_device = device;
	_pointsPerThread = DEFAULT_POINTS_PER_THREAD;
	_numThreads = DEFAULT_NUM_THREADS;
	_numBlocks = DEFAULT_NUM_BLOCKS;

	if(!(compression == Compression::COMPRESSED || compression == Compression::UNCOMPRESSED || compression == Compression::BOTH)) {
		throw KeyFinderException("Invalid argument for compression");
	}
	_compression = compression;

	if(threads != 0) {
		_numThreads = threads;
	}

	if(blocks != 0) {
		_numBlocks = blocks;
	}

	if(pointsPerThread != 0) {
		_pointsPerThread = pointsPerThread;
	}

	if(start.cmp(secp256k1::N) >= 0) {
		throw KeyFinderException("Starting key is out of range");
	}

	if(targetHashes.size() == 0) {
		throw KeyFinderException("Requires at least 1 target");
	}

	// Convert each address from base58 encoded form to a 160-bit integer
	for(unsigned int i = 0; i < targetHashes.size(); i++) {
		KeyFinderTarget t;

		if(!Address::verifyAddress(targetHashes[i])) {
			throw KeyFinderException("Invalid address");
		}

		Base58::toHash160(targetHashes[i], t.hash);

		_targets.push_back(t);
	}

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

void KeyFinder::setTargetHashes()
{
	// Set the target in constant memory
	std::vector<struct hash160> targets;
	for(int i = 0; i < _targets.size(); i++) {
		struct hash160 h;
		memcpy(h.h, _targets[i].hash, sizeof(unsigned int) * 5);
		targets.push_back(h);
	}

	cudaError_t err = setTargetHash(targets);
	if(err) {
		std::string cudaErrorString(cudaGetErrorString(err));

		throw KeyFinderException("Error initializing device: " + cudaErrorString);
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

	setTargetHashes();

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

	for(unsigned long long i = 0; i < _startingPoints.size(); i++) {
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

void KeyFinder::removeHashFromList(const unsigned int hash[5])
{
	for(std::vector<KeyFinderTarget>::iterator i = _targets.begin(); i != _targets.end(); ++i) {
		if(memcmp((*i).hash, hash, sizeof(unsigned int) * 5) == 0) {
			_targets.erase(i);
			break;
		}
	}
}

bool KeyFinder::isHashInList(const unsigned int hash[5])
{
	for(std::vector<KeyFinderTarget>::iterator i = _targets.begin(); i != _targets.end(); ++i) {
		if(memcmp((*i).hash, hash, sizeof(unsigned int) * 5) == 0) {
			return true;
		}
	}

	return false;
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
		if(!isHashInList(rPtr->digest)) {
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
			_statusCallback(info);

			timer.start();
			prevIterCount = _iterCount;
			_totalTime += t;
		}


		// Report any results
		if(_devCtx->resultFound()) {
			std::vector<KeyFinderResult> results;

			getResults(results);

			for(unsigned int i = 0; i < results.size(); i++) {
				unsigned int index = _devCtx->getIndex(results[i].block, results[i].thread, results[i].index);

				secp256k1::uint256 exp = _exponents[index];
				secp256k1::ecpoint publicKey = results[i].p;

				unsigned long long offset = (unsigned long long)_numBlocks * _numThreads * _pointsPerThread * _iterCount;
				exp = secp256k1::addModN(exp, secp256k1::uint256(offset));

				if(!verifyKey(exp, publicKey, results[i].hash, results[0].compressed)) {
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
			for(int i = 0; i < results.size(); i++) {
				removeHashFromList(results[i].hash);
			}

			// Update hash targets on device
			setTargetHashes();

			//_running = false;
		}
		_iterCount++;

		if((_range > 0 && _iterCount * pointsPerIteration >= _range) || _targets.size() == 0) {
			_running = false;
		}
	}
}