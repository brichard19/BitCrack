#ifndef _ADDRESS_MINER
#define _ADDRESS_MINER

#include <vector>
#include "secp256k1.h"

//#include "DeviceContext.h"
//
//#include "AddressMinerShared.h"

typedef struct {
	secp256k1::ecpoint publicKey;
	secp256k1::uint256 privateKey;
	std::string pattern;
	bool compressed;
}AddressMinerResultInfo;



typedef struct {
	int device;
	double speed;
	unsigned long long total;
	unsigned int totalTime;
}AddressMinerStatusInfo;


extern class DeviceContext;

class AddressMinerException {

public:

	AddressMinerException(const std::string &msg)
	{
		this->msg = msg;
	}

	std::string msg;
};


class AddressMiner {

private:

	unsigned int _statusInterval;

	DeviceContext *_devCtx;

	unsigned long long _iterCount;
	unsigned long long _total;
	unsigned int _totalTime;

	// Search pattern
	std::string _pattern;

	// CUDA blocks and threads
	int _numThreads;
	int _numBlocks;
	int _pointsPerThread;
	int _device;

	// Public key point
	secp256k1::ecpoint _point;

	// Exponent/point pairs
	std::vector<secp256k1::uint256> _exponents;
	std::vector<secp256k1::ecpoint> _startingPoints;


	// Each index of each thread gets a flag to indicate if it found a valid hash
	unsigned int *_hashFoundFlags;

	bool _running;

	void(*_resultCallback)(AddressMinerResultInfo);
	void(*_statusCallback)(AddressMinerStatusInfo);
	
	
	static void defaultResultCallback(AddressMinerResultInfo result);
	static void defaultStatusCallback(AddressMinerStatusInfo status);


	void generateStartingPoints();

	bool verifyKey(const secp256k1::ecpoint &startPoint, const secp256k1::uint256 &exponent, const secp256k1::ecpoint &endPoint);

	void applyAutomorphism(const secp256k1::uint256 &k, const secp256k1::ecpoint &p, int autoType, secp256k1::ecpoint &newPoint, secp256k1::uint256 &newK);

	void setAddressMinerTarget(const secp256k1::uint256 &minTarget, const secp256k1::uint256 &maxTarget, const secp256k1::ecpoint &q);

public:

	AddressMiner(const secp256k1::ecpoint &p, const std::string &pattern, int blocks = -1, int threads = -1, int pointsPerThread = -1);
	~AddressMiner();

	void init();
	void run();
	void stop();

	void setResultCallback(void(*callback)(AddressMinerResultInfo));
	void setStatusCallback(void(*callback)(AddressMinerStatusInfo));
	void setStatusInterval(unsigned int interval);
};

#endif