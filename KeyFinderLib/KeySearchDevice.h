#ifndef _KEY_SEARCH_DEVICE_H
#define _KEY_SEARCH_DEVICE_H

#include <vector>
#include <set>
#include "secp256k1.h"
#include "KeySearchTypes.h"


class KeySearchException {

public:

    KeySearchException()
    {

    }

    KeySearchException(const std::string &msg)
    {
        this->msg = msg;
    }

    std::string msg;
};


typedef struct {
    std::string address;
    secp256k1::ecpoint publicKey;
    secp256k1::uint256 privateKey;
    unsigned int hash[5];
    bool compressed;
}KeySearchResult;

// Pure virtual class representing a device that performs a key search
class KeySearchDevice {

public:

    // Initialize the device
    virtual void init(const secp256k1::uint256 &start, int compression) = 0;

    // Perform one iteration
    virtual void doStep() = 0;

    // Tell the device which addresses to search for
    virtual void setTargets(const std::set<KeySearchTarget> &targets) = 0;

    // Get the private keys that have been found so far
    virtual size_t getResults(std::vector<KeySearchResult> &results) = 0;

    // The number of keys searched at each step
    virtual uint64_t keysPerStep() = 0;

    // The name of the device
    virtual std::string getDeviceName() = 0;

    // Memory information for this device
    virtual void getMemoryInfo(uint64_t &freeMem, uint64_t &totalMem) = 0;
};

#endif