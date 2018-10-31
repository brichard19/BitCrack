#ifndef _UTIL_H
#define _UTIL_H

#include <string>
#include <vector>
#include <stdint.h>

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

std::string formatThousands(uint64_t x);
std::string formatSeconds(unsigned int seconds);

uint32_t parseUInt32(std::string s);
uint64_t parseUInt64(std::string s);
bool isHex(const std::string &s);
bool appendToFile(const std::string &fileName, const std::string &s);
bool readLinesFromStream(std::istream &in, std::vector<std::string> &lines);
bool readLinesFromStream(const std::string &fileName, std::vector<std::string> &lines);

std::string format(const char *formatStr, double value);
std::string format(uint32_t value);
std::string format(uint64_t value);
std::string format(int value);
void removeNewline(std::string &s);
unsigned int endian(unsigned int x);

std::string getExeDirectory();
}

#endif