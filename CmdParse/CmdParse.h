#ifndef CMD_PARSE_H
#define CMD_PARSE_H

#include <string>
#include <vector>

class OptArg {

public:

	std::string option;
	std::string arg;

	bool equals(std::string shortForm, std::string longForm = "")
	{
		return (shortForm.length() > 0 && option == shortForm) || option == longForm;
	}
};

class ArgType {

public:
	std::string shortForm;
	std::string longForm;
	bool hasArg;

};

class CmdParse {

private:

	std::vector<ArgType> _argType;

	std::vector<OptArg> _optArgs;

	std::vector<std::string> _operands;

	bool get(const std::string opt, ArgType &t);

public:

	CmdParse();

	void parse(int argc, char **argv);

	void add(const std::string shortForm, const std::string longForm, bool hasArg);
	
	void add(const std::string shortForm, bool hasArg);

	std::vector<OptArg> getArgs();

	std::vector<std::string> getOperands();
};

#endif
