#ifndef _CRYPTO_UTIL_H

namespace crypto {

	class Rng {
		unsigned int _state[16];
		unsigned int _counter;

		void reseed();

	public:
		Rng();
		void get(unsigned char *buf, size_t len);
	};


	void ripemd160(unsigned int *msg, unsigned int *digest);

	void sha256Init(unsigned int *digest);
	void sha256(unsigned int *msg, unsigned int *digest);

	unsigned int checksum(const unsigned int *hash);
}

#endif
