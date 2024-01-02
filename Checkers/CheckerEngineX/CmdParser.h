// Parsing the command line
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <variant>
#include <vector>

std::vector<std::string> split_string(std::string input, std::string delim);
std::vector<std::string> split_string(std::string input,
                                      std::vector<char> delims);
class CmdParser {

public:
  const int arg_length;
  const char **args;
  using CmdType = std::variant<bool, double, int, std::string,
                               std::vector<double>, std::vector<int>,
                               std::vector<std::string>, std::vector<bool>>;
  std::map<std::string, CmdType> options;

  template <typename T> auto convert(std::string arg) {
    if constexpr (std::is_same_v<int, T>) {
      return std::stoi(arg);
    }
    if constexpr (std::is_same_v<double, T>) {
      return std::stod(arg);
    }
    if constexpr (!std::is_same_v<int, T> && !std::is_same_v<double, T>) {
      return std::string(arg);
    }
  };

  template <typename T> auto assign_values(std::vector<std::string> args) {
    auto opt_name = args[0];
    if (args.size() > 2) {
      std::vector<T> converted;

      for (auto i = 1; i < args.size(); ++i) {
        converted.emplace_back(convert<T>(args[i]));
      }
      options[opt_name] = converted;

    } else {
      options[opt_name] = convert<T>(args[1]);
    }
  };

  void add_option(std::string arg_list);

public:
  CmdParser(const int argl, const char **arg) : arg_length(argl), args(arg) {}

  void parse_command_line();

  template <typename T> auto as(std::string option_name) -> decltype(auto) {
    return std::get<T>(options[option_name]);
  }

  bool has_option(std::string option_name) {
    return options.find(option_name) != options.end();
  }

  size_t num_options() const;

  /*
      template<typename... T> bool has_option(T... args){
          //same version as above but for multiple args
          return true;
      } */
};
