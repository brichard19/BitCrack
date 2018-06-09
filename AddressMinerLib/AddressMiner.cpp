#include<stdio.h>

#include "AddressMiner.h"

#include "util.h"
#include "AddressUtil.h"

#include "cudabridge.h"

#include "DeviceContext.h"
//
#include "AddressMinerShared.h"

static const int DEFAULT_POINTS_PER_THREAD = 1;
static const int DEFAULT_NUM_THREADS = 32;
static const int DEFAULT_NUM_BLOCKS = 1;


void AddressMiner::defaultResultCallback(AddressMinerResultInfo result)
{
	// Do nothing
}

void AddressMiner::defaultStatusCallback(AddressMinerStatusInfo status)
{
	// Do nothing
}

AddressMiner::AddressMiner(const secp256k1::ecpoint &point, const std::string &pattern, int blocks, int threads, int pointsPerThread)
{
	_devCtx = NULL;

	_total = 0;
	_statusInterval = 1000;
	_device = 0;
	_pointsPerThread = DEFAULT_POINTS_PER_THREAD;
	_numThreads = DEFAULT_NUM_THREADS;
	_numBlocks = DEFAULT_NUM_BLOCKS;

	if(threads != -1) {
		_numThreads = threads;
	}

	if(blocks != -1) {
		_numBlocks = blocks;
	}

	if(pointsPerThread != -1) {
		_pointsPerThread = pointsPerThread;
	}

	if(pattern.length() == 0) {
		throw AddressMinerException("Invalid pattern: " + pattern);
	}

	if(!secp256k1::pointExists(point)) {
		throw AddressMinerException("Public key does not exist");
	}

	_point = point;

	_statusCallback = NULL;
	_resultCallback = NULL;

	_pattern = pattern;
}

void AddressMiner::setResultCallback(void(*callback)(AddressMinerResultInfo))
{
	_resultCallback = callback;
}

void AddressMiner::setStatusCallback(void(*callback)(AddressMinerStatusInfo))
{
	_statusCallback = callback;
}

void AddressMiner::setStatusInterval(unsigned int interval)
{
	_statusInterval = interval;
}

AddressMiner::~AddressMiner()
{
	if(_devCtx) {
		delete _devCtx;
	}
}

void AddressMiner::init()
{
	if(_devCtx) {
		delete _devCtx;
	}

	_devCtx = new DeviceContext;
	_devCtx->init(_device, _numThreads, _numBlocks, _pointsPerThread);

	secp256k1::uint256 minValue;
	secp256k1::uint256 maxValue;

	Base58::getMinMaxFromPrefix(_pattern, minValue, maxValue);

	setAddressMinerTarget(minValue, maxValue, _point);

	generateStartingPoints();
	_devCtx->copyPoints(_startingPoints);
}

void AddressMiner::generateStartingPoints()
{
	_exponents.clear();

	_startingPoints.clear();
	_iterCount = 0;

	unsigned int totalPoints = _pointsPerThread * _numThreads * _numBlocks;

	secp256k1::generateKeypairsBulk(totalPoints, _point, _exponents, _startingPoints);
}



bool AddressMiner::verifyKey(const secp256k1::ecpoint &startPoint, const secp256k1::uint256 &exponent, const secp256k1::ecpoint &endPoint)
{
	if(!secp256k1::pointExists(endPoint)) {
		return false;
	}

	secp256k1::ecpoint p1 = secp256k1::multiplyPoint(exponent, startPoint);

	return (secp256k1::pointExists(p1)) && p1 == endPoint;
}

/**
 *Given a point and scalar value, computes the new point and new scalar value using
 one of the 6 automorphisms
 */
void AddressMiner::applyAutomorphism(const secp256k1::uint256 &k, const secp256k1::ecpoint &point, int autoType, secp256k1::ecpoint &newPoint, secp256k1::uint256 &newK)
{
	secp256k1::uint256 beta = secp256k1::BETA;
	secp256k1::uint256 beta2 = secp256k1::multiplyModP(beta, beta);

	secp256k1::uint256 lambda = secp256k1::LAMBDA;
	secp256k1::uint256 lambda2 = secp256k1::multiplyModN(lambda, lambda);

	switch(autoType) {
		case AutomorphismType::NONE:
			newK = k;
			newPoint = point;
			break;
		case AutomorphismType::NEGATIVE:
			newK = secp256k1::negModN(k);
			newPoint = point;
			newPoint.y = secp256k1::negModP(newPoint.y);
			break;
		case AutomorphismType::TYPE1:
			newK = secp256k1::multiplyModN(k, lambda);
			newPoint = point;
			newPoint.x = secp256k1::multiplyModP(beta, newPoint.x);
			break;
		case AutomorphismType::TYPE1_NEGATIVE:
			newK = secp256k1::negModN(secp256k1::multiplyModN(k, lambda));
			newPoint = point;
			newPoint.x = secp256k1::multiplyModP(beta, newPoint.x);
			newPoint.y = secp256k1::negModP(newPoint.y);
			break;
		case AutomorphismType::TYPE2:
			newK = secp256k1::multiplyModN(k, lambda2);
			newPoint = point;
			newPoint.x = secp256k1::multiplyModP(beta2, newPoint.x);
			break;
		case AutomorphismType::TYPE2_NEGATIVE:
			newK = secp256k1::negModN(secp256k1::multiplyModN(k, lambda2));
			newPoint = point;
			newPoint.x = secp256k1::multiplyModP(beta2, newPoint.x);
			newPoint.y = secp256k1::negModP(newPoint.y);
			break;
		default:
			throw new AddressMinerException("INVALID AUTOMORPHISM");
	}
}

void AddressMiner::stop()
{
	_running = false;
}

void AddressMiner::run()
{
	_running = true;
	util::Timer timer;
	
	timer.start();
	unsigned long long prevIterCount = 0;
	_totalTime = 0;

	while(_running) {

		//_devCtx.doStep();
		KernelParams params = _devCtx->getKernelParams();

		callAddressMinerKernel(params);

		// Update status
		unsigned int t = timer.getTime();
		if(t >= _statusInterval) {

			AddressMinerStatusInfo info;
			unsigned long long count = 12 * (_iterCount - prevIterCount) * _numBlocks * _numThreads * _pointsPerThread;
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

		_iterCount++;

		// Report any results
		if (_devCtx->resultFound()) {
			std::vector<AddressMinerResult> results;

			_devCtx->getAddressMinerResults(results);


			for(unsigned int i = 0; i < results.size(); i++) {

				int index = _devCtx->getIndex(results[i].block, results[i].thread, results[i].index);

				secp256k1::ecpoint p = results[i].p;

				if(!secp256k1::pointExists(p)) {
					throw AddressMinerException("POINT DOES NOT EXIST");
				}

				AddressMinerResultInfo args;

				secp256k1::uint256 k = _exponents[index];
				secp256k1::uint256 exp = secp256k1::addModN(k, secp256k1::uint256(_iterCount));

				secp256k1::uint256 newExp = exp;
				secp256k1::ecpoint newPoint;

				applyAutomorphism(exp, p, results[i].autoType, newPoint, newExp);

				if(!verifyKey(_point, newExp, p)) {
					printf("exp:%s\n", newExp.toString().c_str());
					printf("x:%s\ny:%s\n", p.x.toString().c_str(), p.y.toString().c_str());
					throw AddressMinerException("ERROR: Key validation failed!");
				}

				args.privateKey = newExp;
				args.publicKey = p;
				args.pattern = _pattern;
				args.compressed = results[i].compressed;
				_resultCallback(args);
			}

			_running = false;
		}
	}
}


void AddressMiner::setAddressMinerTarget(const secp256k1::uint256 &minTarget, const secp256k1::uint256 &maxTarget, const secp256k1::ecpoint &q)
{
	unsigned int minWords[6] = { 0 };
	unsigned int maxWords[6] = { 0 };
	unsigned int qxWords[8] = { 0 };
	unsigned int qyWords[8] = { 0 };

	minTarget.exportWords(minWords, 6, secp256k1::uint256::BigEndian);

	maxTarget.exportWords(maxWords, 6, secp256k1::uint256::BigEndian);

	q.x.exportWords(qxWords, 8, secp256k1::uint256::BigEndian);
	q.y.exportWords(qyWords, 8, secp256k1::uint256::BigEndian);


	cudaError_t err = setMinMaxTarget(minWords, maxWords, qxWords, qyWords);

	if(err != cudaSuccess) {
		throw AddressMinerException("Error setting target on device");
	}
}