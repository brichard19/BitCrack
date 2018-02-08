#ifndef _UTIL_H
#define _UTIL_H

#include<string>

namespace util {

class Timer {

private:
    unsigned int _startTime;

public:
    Timer();
    void start();
    unsigned int getTime();
};

unsigned int getSystemTime();
void sleep(int seconds);
std::string formatThousands(unsigned long long x);

unsigned int parseUInt32(std::string s);
unsigned long long parseUInt64(std::string s);
bool isHex(const std::string &s);

void secureRandom(unsigned char *buf, unsigned int count);
}

#endif