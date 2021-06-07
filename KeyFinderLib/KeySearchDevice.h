#ifndef KEY_SEARCH_DEVICE_H
#define KEY_SEARCH_DEVICE_H

#include <vector>
#include <set>
#include "secp256k1.h"
#include "KeySearchTypes.h"


class KeySearchException {

public:

    KeySearchException()
    {

    }

    KeySearchException(const std::string &msg, const std::string &description)
    {
        this->msg = msg;
        this->description = description;
    }

    std::string msg;
    std::string description;
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

    virtual ~KeySearchDevice() {};

    // Initialize the device
    virtual void init(const secp256k1::uint256 &start, int compression, const secp256k1::uint256 &stride) = 0;

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

    virtual secp256k1::uint256 getNextKey() = 0;
};

#endif
