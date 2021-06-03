#include <stdio.h>
#include <time.h>

#include "Logger.h"
#include "util.h"

inline tm localtime_xp(time_t timer)
{
	tm bt;
#if defined(__unix__)
	localtime_r(&timer, &bt);
#elif defined(_MSC_VER)
	localtime_s(&bt, &timer);
#else
	static std::mutex mtx;
	std::lock_guard<std::mutex> lock(mtx);
	bt = *std::localtime(&timer);
#endif
	return bt;
}

bool LogLevel::isValid(int level)
{
	switch(level) {
		case Info:
		case Error:
		case Debug:
		case Warning:
		case Notify:
			return true;
		default:
			return false;
	}
}

std::string LogLevel::toString(int level)
{
	switch(level) {
		case Info:
			return "Info";
		case Error:
			return "Error";
		case Debug:
			return "Debug";
        case Warning:
            return "Warning";
		case Notify:
			return "Notify";
		default:
			return "";
	}
}

std::string Logger::getDateTimeString()
{
	time_t     now = time(0);
	struct tm  tstruct;
	char       buf[80];
	tstruct = localtime_xp(now);

	strftime(buf, sizeof(buf), "%Y-%m-%d.%X", &tstruct);

	return std::string(buf);
}

std::string Logger::formatLog(LogLevel::Level logLevel, std::string msg)
{
	std::string dateTime = getDateTimeString();

	std::string prefix = "[" + dateTime + "] [" + LogLevel::toString(logLevel) + "] ";

	std::string padding(prefix.length(), ' ');

	if(msg.find('\n', 0) != std::string::npos) {
 		size_t pos = 0;
		size_t prev = 0;

		while((pos = msg.find('\n', prev)) != std::string::npos) {
			prefix += msg.substr(prev, pos - prev) + "\n" + padding;
			prev = pos + 1;
		}

		prefix += msg.substr(prev);
	} else {
		prefix += msg;
	}

	return prefix;
}

void Logger::log(LogLevel::Level level, std::string msg)
{
	std::string str = formatLog(level, msg);
	if (level == LogLevel::Level::Notify) {
		fprintf(stderr, "\a");
	}
	fprintf(stderr, "%s\n", str.c_str());
}
