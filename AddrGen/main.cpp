#include <iostream>
#include<string>
#include "secp256k1.h"
#include "util.h"
#include "AddressUtil.h"
#include "CmdParse.h"

int main(int argc, char **argv)
{
	bool compressed = true;

	secp256k1::uint256 k;

	k = secp256k1::generatePrivateKey();

	CmdParse parser;

	parser.add("-c", "--compressed", false);
	parser.add("-u", "--uncompressed", false);

	parser.parse(argc, argv);

	std::vector<OptArg> args = parser.getArgs();

	for(unsigned int i = 0; i < args.size(); i++) {
		OptArg arg = args[i];
		
		if(arg.equals("-c", "--compressed")) {
			compressed = true;
		} else if(arg.equals("-u", "--uncompressed")) {
			compressed = false;
		}
	}

	std::vector<std::string> operands = parser.getOperands();

	if(operands.size() > 0) {
		try {
			k = secp256k1::uint256(operands[0]);
		} catch(std::string Err) {
			printf("Error parsing private key: %s\n", Err.c_str());
			return 1;
		}
	}

	if(k.isZero() || k.cmp(secp256k1::N) >= 0)
	{
		printf("Error parsing private key: Private key is out of range\n");

		return 1;
	}

	secp256k1::ecpoint p = secp256k1::multiplyPoint(k, secp256k1::G());
	std::string address = Address::fromPublicKey(p, compressed);

	std::cout << k.toString() << std::endl;
	std::cout << p.toString() << std::endl;
	std::cout << address << std::endl;

	return 0;
}