#include <stdio.h>

#include "KeyFinder.h"
#include "AddressUtil.h"
#include "util.h"
#include "secp256k1.h"
#include "CmdParse.h"
#include "cudaUtil.h"


/**
* Callback to display the private key
*/
void resultCallback(KeyFinderResultInfo info)
{
	printf("\n");
	printf("Private key: %s\n", info.privateKey.toString(16).c_str());
	printf("Public key:  %s\n", info.publicKey.x.toString(16).c_str());
	printf("             %s\n", info.publicKey.y.toString(16).c_str());
	printf("\n");

}

/**
Callback to display progress
*/
void statusCallback(KeyFinderStatusInfo info)
{
	if(info.speed < 0.01) {
		printf("\r< 0.01 MKey/s (%s total) [%s]", util::formatThousands(info.total).c_str(), util::formatSeconds((unsigned int)(info.totalTime/1000)).c_str());
	} else {
		printf("\r%.2f MKey/s (%s total) [%s]", info.speed, util::formatThousands(info.total).c_str(), util::formatSeconds((unsigned int)(info.totalTime/1000)).c_str());
	}
}


void usage()
{
	printf("[OPTIONS] TARGET\n");
	printf("Where TARGET is an address\n\n");

	printf("Integer arguments can be in decimal (e.g. 123) or hex (e.g. 0x7B or 7Bh)\n\n");
	
	printf("-c, --compressed        Compressed points\n");
	printf("-u, --uncompressed      Uncompressed points\n");
	printf("-d, --device            The device to use\n");
	printf("-b, --blocks            Number of blocks\n");
	printf("-t, --threads           Threads per block\n");
	printf("-p, --per-thread        Keys per thread\n");
	printf("-s, --start             Staring key, in hex\n");
	printf("-r, --range             Number of keys to search\n");
}


/**
 Finds default parameters depending on the device
 */
typedef struct {
	int threads;
	int blocks;
	int pointsPerThread;
}DeviceParameters;

DeviceParameters findDefaultParameters(int device)
{
	cuda::CudaDeviceInfo devInfo = cuda::getDeviceInfo(device);

	DeviceParameters p;
	p.threads = 256;
	p.blocks = devInfo.mpCount * 16;
	p.pointsPerThread = 32;

	return p;
}




int main(int argc, char **argv)
{
	int device = 0;
	int threads = 0;
	int blocks = 0;
	int pointsPerThread = 0;
	int compression = KeyFinder::Compression::COMPRESSED;

	bool optCompressed = false;
	bool optUncompressed = false;

	std::vector<std::string> targetList;
	secp256k1::uint256 start(1);
	unsigned long long range = 0;

	if(cuda::getDeviceCount == 0) {
		printf("No CUDA devices available\n");
		return 1;
	}

	if(argc == 1) {
		usage();
		return 0;
	}

	CmdParse parser;
	parser.add("-d", "--device", true);
	parser.add("-t", "--threads", true);
	parser.add("-b", "--blocks", true);
	parser.add("-p", "--per-thread", true);
	parser.add("-s", "--start", true);
	parser.add("-r", "--range", true);
	parser.add("-d", "--device", true);
	parser.add("-c", "--compressed", false);
	parser.add("-u", "--uncompressed", false);

	parser.parse(argc, argv);
	std::vector<OptArg> args = parser.getArgs();

	for(unsigned int i = 0; i < args.size(); i++) {
		OptArg optArg = args[i];
		std::string opt = args[i].option;

		try {
			if(optArg.equals("-t", "--threads")) {
				threads = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-b", "--blocks")) {
				blocks = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-p", "--points")) {
				pointsPerThread = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-s", "--start")) {
				start = secp256k1::uint256(optArg.arg);
				if(start.cmp(secp256k1::N) >= 0) {
					throw std::string("argument is out of range");
				}
			} else if(optArg.equals("-r", "--range")) {
				range = util::parseUInt64(optArg.arg);
			} else if(optArg.equals("-p", "--per-thread")) {
				pointsPerThread = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-d", "--device")) {
				device = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-c", "--compressed")) {
				optCompressed = true;
			} else if(optArg.equals("-u", "--uncompressed")) {
				optUncompressed = true;
			}
		} catch(std::string err) {
			printf("Error %s: %s\n", opt.c_str(), err.c_str());
			return 1;
		}
	}

	if(device >= cuda::getDeviceCount()) {
		printf("CUDA device %d does not exist\n", device);
		return 1;
	}

	std::vector<std::string> ops = parser.getOperands();

	if(ops.size() == 0) {
		printf("Missing argument\n");
		usage();
		return 1;
	}

	DeviceParameters devParams = findDefaultParameters(device);
	if(threads == 0) {
		threads = devParams.threads;
	}

	if(blocks == 0) {
		blocks = devParams.blocks;
	}

	if(pointsPerThread == 0) {
		pointsPerThread = devParams.pointsPerThread;
	}
	
	if(optCompressed && optUncompressed) {
		compression = KeyFinder::Compression::BOTH;
	} else if(optCompressed) {
		compression = KeyFinder::Compression::COMPRESSED;
	} else if(optUncompressed) {
		compression = KeyFinder::Compression::UNCOMPRESSED;
	}

	targetList.push_back(ops[0]);

	cuda::CudaDeviceInfo devInfo;
	
	try {
		devInfo = cuda::getDeviceInfo(device);
	} catch(cuda::CudaException &Ex) {
		printf("Error initializing device: %s\n", Ex.msg.c_str());
		return 1;
	}

	printf("Device: %s\n", devInfo.name.c_str());
	printf("Target: %s\n", targetList[0].c_str());

	const char *compStr;
	switch(compression) {
	case KeyFinder::Compression::BOTH:
		compStr = "both";
		break;
	case KeyFinder::Compression::UNCOMPRESSED:
		compStr = "off";
		break;
	case KeyFinder::Compression::COMPRESSED:
		compStr = "on";
	}

	printf("Compression: %s\n", compStr);

	printf("Starting at: %s\n", start.toString().c_str());

	try {
		KeyFinder f(device, start, range, targetList, compression, blocks, threads, pointsPerThread);

		f.setResultCallback(resultCallback);
		f.setStatusInterval(1800);
		f.setStatusCallback(statusCallback);

		printf("Initializing...\n");
		f.init();
		printf("Running\n");
		f.run();
	} catch(KeyFinderException ex) {
		printf("Error: %s\n", ex.msg.c_str());

		return 1;
	}

	return 0;
}