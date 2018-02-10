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


KeyFinder::KeyFinder(const secp256k1::uint256 &start, unsigned long long range, std::vector<std::string> &targetHashes, int blocks, int threads, int pointsPerThread)
{
	_devCtx = NULL;
	_total = 0;
	_statusInterval = 1000;
	_device = 0;
	_pointsPerThread = DEFAULT_POINTS_PER_THREAD;
	_numThreads = DEFAULT_NUM_THREADS;
	_numBlocks = DEFAULT_NUM_BLOCKS;

	if(threads != 0) {
		_numThreads = threads;
	}

	if(blocks != 0) {
		_numBlocks = blocks;
	}

	if(pointsPerThread != 0) {
		_pointsPerThread = pointsPerThread;
	}

	if(targetHashes.size() == 0) {
		throw KeyFinderException("Requires at least 1 target");
	}

	for(int i = 0; i < targetHashes.size(); i++) {
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


void KeyFinder::init()
{
	// Allocate device memory
	_devCtx = new DeviceContext;
	_devCtx->init(_device, _numThreads, _numBlocks, _pointsPerThread);

	// Copy points to device
	generateStartingPoints();
	_devCtx->copyPoints(_startingPoints);

	// Set the target
	setTargetHash(_targets[0].hash);

	// Set the incrementor
	secp256k1::ecpoint g = secp256k1::G();
	secp256k1::ecpoint p = secp256k1::multiplyPoint(secp256k1::uint256(_numThreads * _numBlocks * _pointsPerThread), g);
	setIncrementorPoint(p.x, p.y);
}


void KeyFinder::generateStartingPoints()
{
	_exponents.clear();
	_startingPoints.clear();
	_iterCount = 0;

	unsigned int totalPoints = _pointsPerThread * _numThreads * _numBlocks;

	// Generate key pairs for k, k+1, k+2 ... k + <total points in parallel - 1>
	secp256k1::uint256 privKey = _startExponent;

	for(int i = 0; i < totalPoints; i++) {
		_exponents.push_back(privKey.add(i));
	}

	secp256k1::generateKeyPairsBulk(secp256k1::G(), _exponents, _startingPoints);
}


void KeyFinder::stop()
{
	_running = false;
}

/**
 Verified this private key produces this public key and hash
 */
bool KeyFinder::verifyKey(const secp256k1::uint256 &privateKey, const secp256k1::ecpoint &publicKey, const unsigned int hash[5])
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
	Hash::hashPublicKey(xWords, yWords, digest);

	for(int i = 0; i < 5; i++) {
		if(digest[i] != hash[i]) {
			return false;
		}
	}
	
	return true;
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
			callKeyFinderKernel(params, true);
		} else {
			callKeyFinderKernel(params, false);
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

			_devCtx->getKeyFinderResults(results);

			for(int i = 0; i < results.size(); i++) {
				unsigned int index = _devCtx->getIndex(results[0].block, results[0].thread, results[0].index);

				secp256k1::uint256 exp = _exponents[index];
				secp256k1::ecpoint publicKey = results[0].p;

				unsigned long long offset = (unsigned long long)_numBlocks * _numThreads * _pointsPerThread * _iterCount;
				exp = secp256k1::addModN(exp, secp256k1::uint256(offset));

				if(!verifyKey(exp, publicKey, results[0].hash)) {
					throw KeyFinderException("Invalid point");
				}

				KeyFinderResultInfo info;
				info.privateKey = exp;
				info.publicKey = publicKey;

				_resultCallback(info);
			}


			_running = false;
		}
		_iterCount++;

		if(_iterCount * pointsPerIteration >= _range) {
			_running = false;
		}
	}
}