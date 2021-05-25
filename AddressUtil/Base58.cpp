#include <map>
#include "CryptoUtil.h"
#include "AddressUtil.h"

struct Base58Map {
	static std::map<char, unsigned int> createBase58OrdinalMap()
	{
		std::map<char, unsigned int> m;

		m.insert(std::pair<char, unsigned int>('1', 0));
		m.insert(std::pair<char, unsigned int>('2', 1));
		m.insert(std::pair<char, unsigned int>('3', 2));
		m.insert(std::pair<char, unsigned int>('4', 3));
		m.insert(std::pair<char, unsigned int>('5', 4));
		m.insert(std::pair<char, unsigned int>('6', 5));
		m.insert(std::pair<char, unsigned int>('7', 6));
		m.insert(std::pair<char, unsigned int>('8', 7));
		m.insert(std::pair<char, unsigned int>('9', 8));
		m.insert(std::pair<char, unsigned int>('A', 9));
		m.insert(std::pair<char, unsigned int>('B', 10));
		m.insert(std::pair<char, unsigned int>('C', 11));
		m.insert(std::pair<char, unsigned int>('D', 12));
		m.insert(std::pair<char, unsigned int>('E', 13));
		m.insert(std::pair<char, unsigned int>('F', 14));
		m.insert(std::pair<char, unsigned int>('G', 15));
		m.insert(std::pair<char, unsigned int>('H', 16));
		m.insert(std::pair<char, unsigned int>('J', 17));
		m.insert(std::pair<char, unsigned int>('K', 18));
		m.insert(std::pair<char, unsigned int>('L', 19));
		m.insert(std::pair<char, unsigned int>('M', 20));
		m.insert(std::pair<char, unsigned int>('N', 21));
		m.insert(std::pair<char, unsigned int>('P', 22));
		m.insert(std::pair<char, unsigned int>('Q', 23));
		m.insert(std::pair<char, unsigned int>('R', 24));
		m.insert(std::pair<char, unsigned int>('S', 25));
		m.insert(std::pair<char, unsigned int>('T', 26));
		m.insert(std::pair<char, unsigned int>('U', 27));
		m.insert(std::pair<char, unsigned int>('V', 28));
		m.insert(std::pair<char, unsigned int>('W', 29));
		m.insert(std::pair<char, unsigned int>('X', 30));
		m.insert(std::pair<char, unsigned int>('Y', 31));
		m.insert(std::pair<char, unsigned int>('Z', 32));
		m.insert(std::pair<char, unsigned int>('a', 33));
		m.insert(std::pair<char, unsigned int>('b', 34));
		m.insert(std::pair<char, unsigned int>('c', 35));
		m.insert(std::pair<char, unsigned int>('d', 36));
		m.insert(std::pair<char, unsigned int>('e', 37));
		m.insert(std::pair<char, unsigned int>('f', 38));
		m.insert(std::pair<char, unsigned int>('g', 39));
		m.insert(std::pair<char, unsigned int>('h', 40));
		m.insert(std::pair<char, unsigned int>('i', 41));
		m.insert(std::pair<char, unsigned int>('j', 42));
		m.insert(std::pair<char, unsigned int>('k', 43));
		m.insert(std::pair<char, unsigned int>('m', 44));
		m.insert(std::pair<char, unsigned int>('n', 45));
		m.insert(std::pair<char, unsigned int>('o', 46));
		m.insert(std::pair<char, unsigned int>('p', 47));
		m.insert(std::pair<char, unsigned int>('q', 48));
		m.insert(std::pair<char, unsigned int>('r', 49));
		m.insert(std::pair<char, unsigned int>('s', 50));
		m.insert(std::pair<char, unsigned int>('t', 51));
		m.insert(std::pair<char, unsigned int>('u', 52));
		m.insert(std::pair<char, unsigned int>('v', 53));
		m.insert(std::pair<char, unsigned int>('w', 54));
		m.insert(std::pair<char, unsigned int>('x', 55));
		m.insert(std::pair<char, unsigned int>('y', 56));
		m.insert(std::pair<char, unsigned int>('z', 57));

		return m;
	}

	static std::map<unsigned int, char> createBase58ReverseMap()
	{
		std::map<unsigned int, char> m;

		m.insert(std::pair<unsigned int, char>( 0, '1'));
		m.insert(std::pair<unsigned int, char>( 1, '2'));
		m.insert(std::pair<unsigned int, char>( 2, '3'));
		m.insert(std::pair<unsigned int, char>( 3, '4'));
		m.insert(std::pair<unsigned int, char>( 4, '5'));
		m.insert(std::pair<unsigned int, char>( 5, '6'));
		m.insert(std::pair<unsigned int, char>( 6, '7'));
		m.insert(std::pair<unsigned int, char>( 7, '8'));
		m.insert(std::pair<unsigned int, char>( 8, '9'));
		m.insert(std::pair<unsigned int, char>( 9, 'A'));
		m.insert(std::pair<unsigned int, char>( 10, 'B'));
		m.insert(std::pair<unsigned int, char>( 11, 'C'));
		m.insert(std::pair<unsigned int, char>( 12, 'D'));
		m.insert(std::pair<unsigned int, char>( 13, 'E'));
		m.insert(std::pair<unsigned int, char>( 14, 'F'));
		m.insert(std::pair<unsigned int, char>( 15, 'G'));
		m.insert(std::pair<unsigned int, char>( 16, 'H'));
		m.insert(std::pair<unsigned int, char>( 17, 'J'));
		m.insert(std::pair<unsigned int, char>( 18, 'K'));
		m.insert(std::pair<unsigned int, char>( 19, 'L'));
		m.insert(std::pair<unsigned int, char>( 20, 'M'));
		m.insert(std::pair<unsigned int, char>( 21, 'N'));
		m.insert(std::pair<unsigned int, char>( 22, 'P'));
		m.insert(std::pair<unsigned int, char>( 23, 'Q'));
		m.insert(std::pair<unsigned int, char>( 24, 'R'));
		m.insert(std::pair<unsigned int, char>( 25, 'S'));
		m.insert(std::pair<unsigned int, char>( 26, 'T'));
		m.insert(std::pair<unsigned int, char>( 27, 'U'));
		m.insert(std::pair<unsigned int, char>( 28, 'V'));
		m.insert(std::pair<unsigned int, char>( 29, 'W'));
		m.insert(std::pair<unsigned int, char>( 30, 'X'));
		m.insert(std::pair<unsigned int, char>( 31, 'Y'));
		m.insert(std::pair<unsigned int, char>( 32, 'Z'));
		m.insert(std::pair<unsigned int, char>( 33, 'a'));
		m.insert(std::pair<unsigned int, char>( 34, 'b'));
		m.insert(std::pair<unsigned int, char>( 35, 'c'));
		m.insert(std::pair<unsigned int, char>( 36, 'd'));
		m.insert(std::pair<unsigned int, char>( 37, 'e'));
		m.insert(std::pair<unsigned int, char>( 38, 'f'));
		m.insert(std::pair<unsigned int, char>( 39, 'g'));
		m.insert(std::pair<unsigned int, char>( 40, 'h'));
		m.insert(std::pair<unsigned int, char>( 41, 'i'));
		m.insert(std::pair<unsigned int, char>( 42, 'j'));
		m.insert(std::pair<unsigned int, char>( 43, 'k'));
		m.insert(std::pair<unsigned int, char>( 44, 'm'));
		m.insert(std::pair<unsigned int, char>( 45, 'n'));
		m.insert(std::pair<unsigned int, char>( 46, 'o'));
		m.insert(std::pair<unsigned int, char>( 47, 'p'));
		m.insert(std::pair<unsigned int, char>( 48, 'q'));
		m.insert(std::pair<unsigned int, char>( 49, 'r'));
		m.insert(std::pair<unsigned int, char>( 50, 's'));
		m.insert(std::pair<unsigned int, char>( 51, 't'));
		m.insert(std::pair<unsigned int, char>( 52, 'u'));
		m.insert(std::pair<unsigned int, char>( 53, 'v'));
		m.insert(std::pair<unsigned int, char>( 54, 'w'));
		m.insert(std::pair<unsigned int, char>( 55, 'x'));
		m.insert(std::pair<unsigned int, char>( 56, 'y'));
		m.insert(std::pair<unsigned int, char>( 57, 'z'));

		return m;
	}

	static std::map<char, unsigned int> ordinal;
	static std::map<unsigned int, char> reverse;
};

std::map<char, unsigned int> Base58Map::ordinal = Base58Map::createBase58OrdinalMap();
std::map<unsigned int, char> Base58Map::reverse = Base58Map::createBase58ReverseMap();

/**
 * Converts a base58 string to uint256
 */
secp256k1::uint256 Base58::toBigInt(const std::string &s)
{
	secp256k1::uint256 value;

	for(size_t i = 0, il = s.length(); i < il; i++) {
		value = value.mul(58).add(Base58Map::ordinal.find(s[i])->second);
	}

	return value;
}

void Base58::toHash160(const std::string &s, unsigned int hash[5])
{
	secp256k1::uint256 value = toBigInt(s);
	unsigned int words[6];

	value.exportWords(words, 6, secp256k1::uint256::BigEndian);

	hash[0] = words[0];
	hash[1] = words[1];
	hash[2] = words[2];
	hash[3] = words[3];
	hash[4] = words[4];
}

bool Base58::isBase58(const std::string &value)
{
	for(size_t i = 0; i < value.length(); i++) {
		if(Base58Map::ordinal.find(value[i]) == Base58Map::ordinal.end()) {
			return false;
		}
	}

	return true;
}

std::string Base58::toBase58(secp256k1::uint256 value)
{
	std::string result;

	for (unsigned int i = 0; i <= 32; i++) {
		result.insert(0, 1, Base58Map::reverse.find(value.mod(58).toInt32())->second);
		value = value.div(58);
	}

	return result;
}

void Base58::getMinMaxFromPrefix(const std::string &prefix, secp256k1::uint256 &minValueOut, secp256k1::uint256 &maxValueOut)
{
	secp256k1::uint256 minValue = toBigInt(prefix);
	secp256k1::uint256 maxValue = minValue;
	int exponent = 1;

	// 2^192
	unsigned int expWords[] = { 0, 0, 0, 0, 0, 0, 1, 0 };

	secp256k1::uint256 exp(expWords);

	// Find the smallest 192-bit number that starts with the prefix. That is, the prefix multiplied
	// by some power of 58
	secp256k1::uint256 nextValue = minValue.mul(58);

	while(nextValue.cmp(exp) < 0) {
		exponent++;
		minValue = nextValue;
		nextValue = nextValue.mul(58);
	}

	secp256k1::uint256 diff = secp256k1::uint256(58).pow(exponent - 1).sub(1);

	maxValue = minValue.add(diff);

	if(maxValue.cmp(exp) > 0) {
		maxValue = exp.sub(1);
	}

	minValueOut = minValue;
	maxValueOut = maxValue;
}
