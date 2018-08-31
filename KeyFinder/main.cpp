#include <stdio.h>
#include <fstream>
#include <iostream>

#include "KeyFinder.h"
#include "AddressUtil.h"
#include "util.h"
#include "secp256k1.h"
#include "CmdParse.h"
#include "cudaUtil.h"
#include "Logger.h"

static std::string _outputFile = "";

/**
* Callback to display the private key
*/
void resultCallback(KeyFinderResultInfo info)
{
	if(_outputFile.length() != 0) {
		Logger::log(LogLevel::Info, "Found key for address '" + info.address + "'. Written to '" + _outputFile + "'");

		std::string s = info.address + " " + info.privateKey.toString(16) + " " + info.publicKey.toString(info.compressed);
		util::appendToFile(_outputFile, s);

		return;
	}

	std::string logStr = "Address:     " + info.address + "\n";
	logStr += "Private key: " + info.privateKey.toString(16) + "\n";
	logStr += "Compressed:  ";

	if(info.compressed) {
		logStr += "yes\n";
	} else {
		logStr += "no\n";
	}

	logStr += "Public key:  \n";

	if(info.compressed) {
		logStr += info.publicKey.toString(true) + "\n";
	} else {
		logStr += info.publicKey.x.toString(16) + "\n";
		logStr += info.publicKey.y.toString(16) + "\n";
	}

	Logger::log(LogLevel::Info, logStr);
}

/**
Callback to display progress
*/
void statusCallback(KeyFinderStatusInfo info)
{
	std::string speedStr;

	if(info.speed < 0.01) {
		speedStr = "< 0.01 MKey/s";
	} else {
		speedStr = util::format("%.2f", info.speed) + " MKey/s";
	}

	std::string totalStr = "(" + util::formatThousands(info.total) + " total)";

	std::string timeStr = "[" + util::formatSeconds((unsigned int)(info.totalTime / 1000)) + "]";

	std::string usedMemStr = util::format((info.deviceMemory - info.freeMemory) / (unsigned long long)(1024 * 1024));

	std::string totalMemStr = util::format(info.deviceMemory / (unsigned long long)(1024 * 1024));

	std::string targetStr = util::format(info.targets) + " target";
	if(info.targets > 1) {
		targetStr += "s";
	}

	// Fit device name in 16 characters, pad with spaces if less
	std::string devName = info.deviceName.substr(0, 16);
	devName += std::string(16 - devName.length(), ' ');

	printf("\r%s %s/%sMB | %s %s %s %s", devName.c_str(), usedMemStr.c_str(), totalMemStr.c_str(), targetStr.c_str(), speedStr.c_str(), totalStr.c_str(), timeStr.c_str());
}

void usage()
{
	printf("[OPTIONS] [TARGETS]\n");
	printf("Where TARGETS is one or more addresses\n\n");

	printf("Integer arguments can be in decimal (e.g. 123) or hex (e.g. 0x7B or 7Bh)\n\n");
	
	printf("-c, --compressed        Compressed points\n");
	printf("-u, --uncompressed      Uncompressed points\n");
	printf("-d, --device            The device to use\n");
	printf("-b, --blocks            Number of blocks\n");
	printf("-t, --threads           Threads per block\n");
	printf("-p, --per-thread        Keys per thread\n");
	printf("-s, --start             Staring key, in hex\n");
	printf("-r, --range             Number of keys to search\n");
	printf("-i, --in                Specify file containing addresses, one per line\n");
	printf("-o, --out               Specify file where results are written\n");
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


static std::string getCompressionString(int mode)
{
	switch(mode) {
	case KeyFinder::Compression::BOTH:
		return "both";
	case KeyFinder::Compression::UNCOMPRESSED:
		return "uncompressed";
	case KeyFinder::Compression::COMPRESSED:
		return "compressed";
	}

	throw std::string("Invalid compression setting");
}

bool readAddressesFromFile(const std::string &fileName, std::vector<std::string> &lines)
{
	if(fileName == "-") {
		return util::readLinesFromStream(std::cin, lines);
	} else {
		return util::readLinesFromStream(fileName, lines);
	}
}

int main(int argc, char **argv)
{
	int device = 0;
	int threads = 0;
	int blocks = 0;
	int pointsPerThread = 0;
	int compression = KeyFinder::Compression::COMPRESSED;
	std::string targetFile = "";

	std::string outputFile = "";

	bool optCompressed = false;
	bool optUncompressed = false;
	
	std::vector<std::string> targetList;
	secp256k1::uint256 start(1);
	unsigned long long range = 0;
    int deviceCount = 0;

    try {
        deviceCount = cuda::getDeviceCount();
        if(deviceCount == 0) {
            Logger::log(LogLevel::Error, "No CUDA devices available");
            return 1;
        }
    } catch(cuda::CudaException ex) {
        Logger::log(LogLevel::Error, "Error detecting CUDA devices: " + ex.msg);
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
	parser.add("-i", "--in", true);
	parser.add("-o", "--out", true);

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
			} else if(optArg.equals("-i", "--in")) {
				targetFile = optArg.arg;
			} else if(optArg.equals("-o", "--out")) {
				_outputFile = optArg.arg;
			}
		} catch(std::string err) {
			Logger::log(LogLevel::Error, "Error " + opt + ": " + err);
			return 1;
		}
	}


	// Verify device exists
	if(device < 0 || device >= deviceCount) {
		Logger::log(LogLevel::Error, "CUDA device " + util::format(device) + " does not exist");
		return 1;
	}

	// Parse operands
	std::vector<std::string> ops = parser.getOperands();

	if(ops.size() == 0) {
		if(targetFile.length() == 0) {
			Logger::log(LogLevel::Error, "Missing arguments");
			usage();
			return 1;
		}
	} else {
		for(unsigned int i = 0; i < ops.size(); i++) {
			targetList.push_back(ops[i]);
		}
	}


	// Get device parameters (blocks, threads, points per thread)
	DeviceParameters devParams = findDefaultParameters(device);

	// Apply defaults if none given
	if(threads == 0) {
		threads = devParams.threads;
	}

	if(blocks == 0) {
		blocks = devParams.blocks;
	}

	if(pointsPerThread == 0) {
		pointsPerThread = devParams.pointsPerThread;
	}
	
	// Check option for compressed, uncompressed, or both
	if(optCompressed && optUncompressed) {
		compression = KeyFinder::Compression::BOTH;
	} else if(optCompressed) {
		compression = KeyFinder::Compression::COMPRESSED;
	} else if(optUncompressed) {
		compression = KeyFinder::Compression::UNCOMPRESSED;
	}

	cuda::CudaDeviceInfo devInfo;
	
	// Initialize the device
	try {
		devInfo = cuda::getDeviceInfo(device);
	} catch(cuda::CudaException &Ex) {
		Logger::log(LogLevel::Error, "Cannot initialize device: " + Ex.msg);
		return 1;
	}

	Logger::log(LogLevel::Info, "Compression: " + getCompressionString(compression));
	Logger::log(LogLevel::Info, "Starting at: " + start.toString());

	try {
		KeyFinder f(device, start, range, compression, blocks, threads, pointsPerThread);

		f.setResultCallback(resultCallback);
		f.setStatusInterval(1800);
		f.setStatusCallback(statusCallback);


		if(!targetFile.empty()) {
			f.setTargets(targetFile);
		} else {
			f.setTargets(targetList);
		}

		f.init();
		f.run();
	} catch(KeyFinderException ex) {
		Logger::log(LogLevel::Info, "Error: " + ex.msg + " Exiting.");
		return 1;
	}

	return 0;
}