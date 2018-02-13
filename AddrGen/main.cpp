#include <iostream>
#include<string>
#include "secp256k1.h"
#include "util.h"
#include "AddressUtil.h"


int main(int argc, char **argv)
{
	bool compressed = false;

	secp256k1::uint256 k;

	k = secp256k1::generatePrivateKey();

	for(int i = 1; i < argc; i++) {
		if(strcmp(argv[i], "-c") == 0) {
			compressed = true;
		} else {
			k = secp256k1::uint256(argv[i]);
		}
	}
	
	secp256k1::ecpoint p = secp256k1::multiplyPoint(k, secp256k1::G());
	std::string address = Address::fromPublicKey(p, compressed);

	std::cout << k.toString() << std::endl;
	std::cout << p.toString() << std::endl;
	std::cout << address << std::endl;

	return 0;
}