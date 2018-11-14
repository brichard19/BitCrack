#include <stdio.h>
#include <fstream>
#include <iostream>

#include "KeyFinder.h"
#include "AddressUtil.h"
#include "util.h"
#include "secp256k1.h"
#include "CmdParse.h"
#include "Logger.h"

#include "DeviceManager.h"

#ifdef BUILD_CUDA
#include "CudaKeySearchDevice.h"
#endif


#ifdef BUILD_OPENCL
#include "CLKeySearchDevice.h"
#endif

typedef struct {

    // Start at lowest private key
    secp256k1::uint256 startKey = 1;

    // End at highest private key
    secp256k1::uint256 endKey = secp256k1::N;

    unsigned int interval = 1800;

    unsigned int threads = 0;
    unsigned int blocks = 0;
    unsigned int pointsPerThread = 0;
    int compression = PointCompressionType::COMPRESSED;

    std::vector<std::string> targets;
    std::string targetsFile = "";

    std::string checkpointFile = "";

    DeviceManager::DeviceInfo device;

    std::string resultsFile = "";

}RunConfig;

static RunConfig _config;



/**
* Callback to display the private key
*/
void resultCallback(KeySearchResult info)
{
	if(_config.resultsFile.length() != 0) {
		Logger::log(LogLevel::Info, "Found key for address '" + info.address + "'. Written to '" + _config.resultsFile + "'");

		std::string s = info.address + " " + info.privateKey.toString(16) + " " + info.publicKey.toString(info.compressed);
		util::appendToFile(_config.resultsFile, s);

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
void statusCallback(KeySearchStatus info)
{
	std::string speedStr;

	if(info.speed < 0.01) {
		speedStr = "< 0.01 MKey/s";
	} else {
		speedStr = util::format("%.2f", info.speed) + " MKey/s";
	}

	std::string totalStr = "(" + util::formatThousands(info.total) + " total)";

	std::string timeStr = "[" + util::formatSeconds((unsigned int)(info.totalTime / 1000)) + "]";

	std::string usedMemStr = util::format((info.deviceMemory - info.freeMemory) /(1024 * 1024));

	std::string totalMemStr = util::format(info.deviceMemory / (1024 * 1024));

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
	printf("-t, --threads           Threads per block\n");
	printf("-p, --per-thread        Keys per thread\n");
	printf("-s, --start             Staring key, in hex\n");
	//printf("-r, --range             Number of keys to search\n");
	printf("-i, --in                Specify file containing addresses, one per line\n");
	printf("-o, --out               Specify file where results are written\n");
    printf("-l, --list-devices      List available devices\n");
}


/**
 Finds default parameters depending on the device
 */
typedef struct {
	int threads;
	int blocks;
	int pointsPerThread;
}DeviceParameters;

DeviceParameters getDefaultParameters(const DeviceManager::DeviceInfo &device)
{
	DeviceParameters p;
	p.threads = 256;
    p.blocks = 32;
	p.pointsPerThread = 32;

	return p;
}


static std::string getCompressionString(int mode)
{
	switch(mode) {
	case PointCompressionType::BOTH:
		return "both";
	case PointCompressionType::UNCOMPRESSED:
		return "uncompressed";
	case PointCompressionType::COMPRESSED:
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

static KeySearchDevice *getDeviceContext(DeviceManager::DeviceInfo &device, int blocks, int threads, int pointsPerThread)
{
#ifdef BUILD_CUDA
    if(device.type == DeviceManager::DeviceType::CUDA) {
        return new CudaKeySearchDevice((int)device.physicalId, threads, pointsPerThread, blocks);
    }
#endif

#ifdef BUILD_OPENCL
    if(device.type == DeviceManager::DeviceType::OpenCL) {
        return new CLKeySearchDevice(device.physicalId, threads, pointsPerThread, blocks);
    }
#endif

    return NULL;
}

static void printDeviceList(const std::vector<DeviceManager::DeviceInfo> &devices)
{
    for(int i = 0; i < devices.size(); i++) {
        printf("ID:     %d\n", devices[i].id);
        printf("Name:   %s\n", devices[i].name.c_str());
        printf("Memory: %lldMB\n", devices[i].memory / (1024 * 1024));
        printf("\n");
    }
}

int run(RunConfig &config)
{
    Logger::log(LogLevel::Info, "Compression: " + getCompressionString(config.compression));
    Logger::log(LogLevel::Info, "Starting at: " + config.startKey.toString());

    try {

        KeySearchDevice *d = getDeviceContext(config.device, config.blocks, config.threads, config.pointsPerThread);

        KeyFinder f(config.startKey, config.endKey, config.compression, d);

        f.setResultCallback(resultCallback);
        f.setStatusInterval(1800);
        f.setStatusCallback(statusCallback);

        f.init();

        if(!config.targetsFile.empty()) {
            f.setTargets(config.targetsFile);
        } else {
            f.setTargets(config.targets);
        }

        f.run();

        delete d;
    } catch(KeySearchException ex) {
        Logger::log(LogLevel::Info, "Error: " + ex.msg + " Exiting.");
        return 1;
    }

    return 0;
}


int main(int argc, char **argv)
{
	int device = 0;
	bool optCompressed = false;
	bool optUncompressed = false;
    bool listDevices = false;

	//uint64_t range = 0;
    int deviceCount = 0;

    std::vector<DeviceManager::DeviceInfo> devices;

    // Check for supported devices
    try {
        devices = DeviceManager::getDevices();

        if(devices.size() == 0) {
            Logger::log(LogLevel::Error, "No devices available");
            return 1;
        }
    } catch(DeviceManager::DeviceManagerException ex) {
        Logger::log(LogLevel::Error, "Error detecting devices: " + ex.msg);
        return 1;
    }

    // Check for arguments
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
    parser.add("-l", "--list-devices", false);

	parser.parse(argc, argv);
	std::vector<OptArg> args = parser.getArgs();

	for(unsigned int i = 0; i < args.size(); i++) {
		OptArg optArg = args[i];
		std::string opt = args[i].option;

		try {
			if(optArg.equals("-t", "--threads")) {
				_config.threads = util::parseUInt32(optArg.arg);
            } else if(optArg.equals("-b", "--blocks")) {
                _config.blocks = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-p", "--points")) {
				_config.pointsPerThread = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-s", "--start")) {
				_config.startKey = secp256k1::uint256(optArg.arg);
				if(_config.startKey.cmp(secp256k1::N) >= 0) {
					throw std::string("argument is out of range");
				}
			}/* else if(optArg.equals("-r", "--range")) {
				range = util::parseUInt64(optArg.arg);
			}*/ else if(optArg.equals("-p", "--per-thread")) {
				_config.pointsPerThread = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-d", "--device")) {
				device = util::parseUInt32(optArg.arg);
			} else if(optArg.equals("-c", "--compressed")) {
				optCompressed = true;
			} else if(optArg.equals("-u", "--uncompressed")) {
				optUncompressed = true;
			} else if(optArg.equals("-i", "--in")) {
				_config.targetsFile = optArg.arg;
			} else if(optArg.equals("-o", "--out")) {
				_config.resultsFile = optArg.arg;
            } else if(optArg.equals("-l", "--list-devices")) {
                listDevices = true;
            }
		} catch(std::string err) {
			Logger::log(LogLevel::Error, "Error " + opt + ": " + err);
			return 1;
		}
	}

    if(listDevices) {
        printDeviceList(devices);
        return 0;
    }

	// Verify device exists
	if(device < 0 || device >= devices.size()) {
		Logger::log(LogLevel::Error, "device " + util::format(device) + " does not exist");
		return 1;
	}

	// Parse operands
	std::vector<std::string> ops = parser.getOperands();

	if(ops.size() == 0) {
		if(_config.targetsFile.length() == 0) {
			Logger::log(LogLevel::Error, "Missing arguments");
			usage();
			return 1;
		}
	} else {
		for(unsigned int i = 0; i < ops.size(); i++) {
			_config.targets.push_back(ops[i]);
		}
	}

    // Set parameters
    DeviceParameters defaultParameters = getDefaultParameters(devices[device]);
    if(_config.blocks == 0) {
        _config.blocks = defaultParameters.blocks;
    }

    if(_config.threads == 0) {
        _config.threads = defaultParameters.threads;
    }

    if(_config.pointsPerThread == 0) {
        _config.pointsPerThread = defaultParameters.pointsPerThread;
    }


	// Check option for compressed, uncompressed, or both
	if(optCompressed && optUncompressed) {
		_config.compression = PointCompressionType::BOTH;
	} else if(optCompressed) {
		_config.compression = PointCompressionType::COMPRESSED;
	} else if(optUncompressed) {
		_config.compression = PointCompressionType::UNCOMPRESSED;
	}


    return run(_config);
}