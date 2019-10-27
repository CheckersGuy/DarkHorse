//
// Created by robin on 24.08.19.
//

#ifndef COMMANDLINE_COMMANDLINE_H
#define COMMANDLINE_COMMANDLINE_H
#include "Utils.h"
#include <map>
#include <iostream>
#include <string>
#include <iterator>

//A very simple commandLine tool
//Doesnt offer much but should be enough for most projects
struct Option {
    std::string optionName;
    std::vector<std::string> options;

    Option(std::string name) : optionName(name) {}

    Option() = default;

    template<class T=std::string>
    std::vector<T> getParams() {
        std::vector<T> result;
        for (auto &word : options) {
            std::istringstream stream(word);
            T value;
            stream >> value;
            result.emplace_back(value);
        }

        return result;
    }

    bool operator==(const Option &other) {
        return other.optionName == this->optionName;
    }

    bool operator!=(const Option &other) {
        return other.optionName != this->optionName;
    }
};

class CommandLine {
private:


    std::map<std::string, Option> options;
public:
    CommandLine() = default;

    void parseCommandLine(int argl, char **argc);

    bool has(const std::string optionName);

    Option &get(const std::string optionName);

    Option &operator[](std::string optionName);

    std::vector<std::string> getOptionNames();
};


#endif //COMMANDLINE_COMMANDLINE_H
