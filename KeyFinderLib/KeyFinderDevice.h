#ifndef _KEY_FINDER_DEVICE_H
#define _KEY_FINDER_DEVICE_H

#include <vector>
#include "secp256k1.h"

struct KeyFinderResult {

    // TODO: Remove these since they are device-specific
    int thread;
    int block;
    int index;

    bool compressed;

    secp256k1::uint256 privateKey;

    secp256k1::ecpoint p;
    unsigned int hash[5];
};

typedef struct {
    std::string address;
    secp256k1::ecpoint publicKey;
    secp256k1::uint256 privateKey;
    bool compressed;
}KeyFinderResultInfo;


class KeyFinderDevice {

public:

    virtual void init(const void *params, const secp256k1::uint256 &start) = 0;

    virtual bool doStep() = 0;

    virtual bool isDone() = 0;

    virtual void setTargets(const std::vector<hash160> &targets) = 0;

    virtual size_t getResults(std::vector<struct KeyFinderResult> &results) = 0;
};

#endif