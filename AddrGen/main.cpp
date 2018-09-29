#include <iostream>
#include<string>
#include "secp256k1.h"
#include "util.h"
#include "AddressUtil.h"
#include "CmdParse.h"

int main(int argc, char **argv)
{
    std::vector<secp256k1::uint256> keys;

	bool compressed = true;
    bool printPrivate = false;
    bool printPublic = false;
    bool printAddr = false;
    bool printAll = true;
    int count = 1;

	secp256k1::uint256 k;

	k = secp256k1::generatePrivateKey();

	CmdParse parser;

	parser.add("-c", "--compressed", false);
	parser.add("-u", "--uncompressed", false);
    parser.add("-p", "--pub", false);
    parser.add("-k", "--priv", false);
    parser.add("-a", "--addr", false);
    parser.add("-n", true);

	parser.parse(argc, argv);

	std::vector<OptArg> args = parser.getArgs();

	for(unsigned int i = 0; i < args.size(); i++) {
		OptArg arg = args[i];
		
		if(arg.equals("-c", "--compressed")) {
			compressed = true;
		} else if(arg.equals("-u", "--uncompressed")) {
			compressed = false;
        } else if(arg.equals("-k", "--priv")) {
            printAll = false;
            printPrivate = true;
        } else if(arg.equals("-p", "--pub")) {
            printAll = false;
            printPublic = true;
        } else if(arg.equals("-a", "--addr")) {
            printAll = false;
            printAddr = true;
        } else if(arg.equals("-n")) {
            count = (int)util::parseUInt32(arg.arg);
        }
	}

	std::vector<std::string> operands = parser.getOperands();

	if(operands.size() > 0) {
        for(int i = 0; i < operands.size(); i++) {
            try {
                keys.push_back(secp256k1::uint256(operands[i]));
            } catch(std::string err) {
                printf("Error parsing private key: %s\n", err.c_str());
                return 1;
            }
        }
	}

    for(int i = 0; i < keys.size(); i++) {
        secp256k1::uint256 k = keys[i];

        if(k.isZero() || k.cmp(secp256k1::N) >= 0)
        {
            printf("Error parsing private key: Private key is out of range\n");

            return 1;
        }

        secp256k1::ecpoint p = secp256k1::multiplyPoint(k, secp256k1::G());
        std::string address = Address::fromPublicKey(p, compressed);

        if(printAll || printPrivate) {
            std::cout << k.toString() << std::endl;
        }
        if(printAll || printPublic) {
            std::cout << p.toString() << std::endl;
        }
        if(printAll || printAddr) {
            std::cout << address << std::endl;
        }
    }

	return 0;
}