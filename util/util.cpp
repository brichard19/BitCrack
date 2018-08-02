#include<stdio.h>
#include<string>
#include<fstream>
#include<vector>
#include<set>

#include"util.h"

#ifdef _WIN32
#include<windows.h>
#else
#include<unistd.h>
#include<sys/stat.h>
#include<sys/time.h>
#endif

namespace util {

    unsigned int getSystemTime()
    {
#ifdef _WIN32
        return GetTickCount();
#else
        struct timeval t;
        gettimeofday(&t, NULL);
        return t.tv_sec * 1000 + t.tv_usec / 1000;
#endif
    }

    Timer::Timer()
    {
        _startTime = 0;
    }

    void Timer::start()
    {
        _startTime = getSystemTime();
    }

    unsigned int Timer::getTime()
    {
        return getSystemTime() - _startTime;
    }

    void sleep(int seconds)
    {
#ifdef _WIN32
        Sleep(seconds * 1000);
#else
        sleep(seconds);
#endif
    }

	std::string formatThousands(unsigned long long x)
	{
		char buf[32] = "";

		sprintf(buf, "%lld", x);

		std::string s(buf);

		int len = (int)s.length();

		int numCommas = (len - 1) / 3;

		if(numCommas == 0) {
			return s;
		}

		std::string result = "";

		int count = ((len % 3) == 0) ? 0 : (3 - (len % 3));

		for(int i = 0; i < len; i++) {
			result += s[i];

			if(count++ == 2 && i < len - 1) {
				result += ",";
				count = 0;
			}
		}

		return result;
	}

	unsigned int parseUInt32(std::string s)
	{
		return (unsigned int)parseUInt64(s);
	}

	unsigned long long parseUInt64(std::string s)
	{
		unsigned long long val = 0;
		bool isHex = false;

		if(s[0] == '0' && s[1] == 'x') {
			isHex = true;
			s = s.substr(2);
		}
		
		if(s[s.length() - 1] == 'h') {
			isHex = true;
			s = s.substr(0, s.length() - 1);
		}

		if(isHex) {
			if(sscanf(s.c_str(), "%llx", &val) != 1) {
				throw std::string("Expected an integer");
			}
		} else {
			if(sscanf(s.c_str(), "%lld", &val) != 1) {
				throw std::string("Expected an integer");
			}
		}

		return val;
	}

	bool isHex(const std::string &s)
	{
		int len = 0;

		for(int i = 0; i < len; i++) {
			char c = s[i];

			if(!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
				return false;
			}
		}

		return true;
	}

	std::string formatSeconds(unsigned int seconds)
	{
		char s[128] = { 0 };

		unsigned int days = seconds / 86400;
		unsigned int hours = (seconds % 86400) / 3600;
		unsigned int minutes = (seconds % 3600) / 60;
		unsigned int sec = seconds % 60;

		if(days > 0) {
			sprintf(s, "%d:%02d:%02d:%02d", days, hours, minutes, sec);
		} else {
			sprintf(s, "%02d:%02d:%02d", hours, minutes, sec);
		}
		

		return std::string(s);
	}

	long getFileSize(const std::string &fileName)
	{
		FILE *fp = fopen(fileName.c_str(), "rb");
		if(fp == NULL) {
			return -1;
		}

		fseek(fp, 0, SEEK_END);

		long pos = ftell(fp);

		fclose(fp);

		return pos;
	}

	bool readLinesFromStream(const std::string &fileName, std::vector<std::string> &lines)
	{
		std::ifstream inFile(fileName.c_str());

		if(!inFile.is_open()) {
			return false;
		}

		return readLinesFromStream(inFile, lines);
	}

	bool readLinesFromStream(std::istream &in, std::vector<std::string> &lines)
	{
		std::string line;

		while(std::getline(in, line)) {
			if(line.length() > 0) {
				lines.push_back(line);
			}
		}

		return true;
	}

	bool appendToFile(const std::string &fileName, const std::string &s)
	{
		std::ofstream outFile;
		bool newline = false;

		if(getFileSize(fileName) > 0) {
			newline = true;
		}

		outFile.open(fileName.c_str(), std::ios::app);

		if(!outFile.is_open()) {
			return false;
		}

		// Add newline following previous line
		if(newline) {
			outFile << std::endl;
		}

		outFile << s;

		return true;
	}

	std::string format(const char *formatStr, double value)
	{
		char buf[100] = { 0 };

		sprintf(buf, formatStr, value);

		return std::string(buf);
	}

	std::string format(unsigned int value)
	{
		char buf[100] = { 0 };

		sprintf(buf, "%d", value);

		return std::string(buf);
	}

	std::string format(unsigned long long value)
	{
		char buf[100] = { 0 };

		sprintf(buf, "%lld", value);

		return std::string(buf);
	}
}