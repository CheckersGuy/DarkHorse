//
// Created by robin on 24.08.19.
//

#include "CommandLine.h"

bool CommandLine::has(const std::string optionName) {
    return (options.find(optionName) != options.end());
}

Option &CommandLine::get(const std::string optionName) {
    return options[optionName];
}

Option &CommandLine::operator[](std::string optionName) {
    return options[optionName];
}

std::vector<std::string> CommandLine::getOptionNames() {
    std::vector<std::string> result;
    for (auto &option : options) {
        result.emplace_back(option.first);
    }
    return result;
}

void CommandLine::parseCommandLine(int argl, char **argc) {
    std::ostringstream stream("");
    for (int i = 1; i < argl; ++i) {
        stream << argc[i] << " ";
    }
    std::vector<std::string> commands = Utils::getDelimiterSeperated(stream.str(), "--");
    for (const auto &word : commands) {
        if (word.empty())
            continue;
        std::vector<std::string> op = Utils::getWhiteSpaceSeperated(word);
        Option option;
        option.optionName = op[0];
        for (int i = 1; i < op.size(); ++i) {
            option.options.emplace_back(op[i]);
        }
        auto pair = std::make_pair(option.optionName, option);
        options.insert(pair);
    }

}