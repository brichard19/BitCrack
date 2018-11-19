#ifndef _CONFIG_FILE_H
#define _CONFIG_FILE_H

#include <string>
#include <map>
#include <vector>
#include <fstream>

class ConfigFileEntry {

public:
    std::string key = "";
    std::string value = "";

    ConfigFileEntry() {}

    ConfigFileEntry(std::string k, std::string v) : key(k), value(v)
    {
    }

};

class ConfigFileReader {

private:
    std::string _path;

    ConfigFileEntry readEntry(const std::string &line);

public:
    ConfigFileReader(const std::string &path);

    bool exists();
    std::map<std::string, ConfigFileEntry> read();
};

#endif