#include <iostream>
#include<string>
#include "secp256k1.h"
#include "util.h"
#include "AddressUtil.h"


int main(int argc, char **argv)
{
	secp256k1::uint256 k;

	if(argc > 1) {
		k = secp256k1::uint256(argv[1]);
	} else {
		k = secp256k1::generatePrivateKey();
	}
	
	secp256k1::ecpoint p = secp256k1::multiplyPoint(k, secp256k1::G());
	std::string address = Address::fromPublicKey(p);

	std::cout << k.toString() << std::endl;
	std::cout << p.toString() << std::endl;
	std::cout << address << std::endl;

	return 0;
}