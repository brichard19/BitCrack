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
std::string formatSeconds(unsigned int seconds);

unsigned int parseUInt32(std::string s);
unsigned long long parseUInt64(std::string s);
bool isHex(const std::string &s);
bool appendToFile(const std::string &fileName, const std::string &s);
bool readLinesFromStream(std::istream &in, std::vector<std::string> &lines);
bool readLinesFromStream(const std::string &fileName, std::vector<std::string> &lines);
unsigned int removeDuplicates(std::vector<std::string> &v);
std::string format(const char *formatStr, double value);
std::string format(unsigned int value);

}

#endif