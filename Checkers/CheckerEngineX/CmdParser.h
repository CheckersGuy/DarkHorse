#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
std::vector<std::string> split(std::string s, std::string delimiter) {
  std::vector<std::string> tokens;
  size_t pos = 0;
  std::string token;
  while ((pos = s.find(delimiter)) != std::string::npos) {
    token = s.substr(0, pos);
    tokens.push_back(token);
    s.erase(0, pos + delimiter.length());
  }
  tokens.push_back(s);

  return tokens;
}

inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
            return !std::isspace(ch);
          }));
}

// trim from end (in place)
inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// trim from both ends (in place)
inline void trim(std::string &s) {
  rtrim(s);
  ltrim(s);
}

template <typename C> struct is_vector : std::false_type {};
template <typename T, typename A>
struct is_vector<std::vector<T, A>> : std::true_type {};
template <typename C> inline constexpr bool is_vector_v = is_vector<C>::value;

struct CmdParser {

public:
  std::unordered_map<std::string, std::vector<std::string>> options;

public:
  void parse(int argl, const char **argc) {
    std::stringstream sstream;
    for (auto i = 1; i < argl; ++i) {
      sstream << argc[i] << " ";
    }

    const auto token_string = sstream.str();

    const auto tokens = split(token_string, "--");

    for (auto i = 0; i < tokens.size(); ++i) {
      const auto temp = tokens[i];
      auto opt = split(temp, " ");
      trim(opt[0]);
      if (opt[0].empty())
        continue;

      options[opt[0]] = std::vector<std::string>{};

      for (auto i = 1; i < opt.size(); ++i) {
        auto value = opt[i];
        trim(value);
        if (value.empty())
          continue;
        options[opt[0]].emplace_back(value);
      }
      for (auto &value : options[opt[0]]) {
        trim(value);
      }
    }
  }

  bool has_option(std::string option_name) {
    return options.find(option_name) != options.end();
  }

  template <typename T> T as(std::string option_name) {
    // support more datatypes later
    auto &args = options[option_name];
    if constexpr (std::is_same_v<int, T>) {
      return std::stoi(args[0]);
    }
    if constexpr (std::is_same_v<std::string, T>) {
      return args[0];
    }

    if constexpr (std::is_same_v<std::vector<int>, T>) {
      std::vector<int> result;
      for (auto value : args) {
        std::cout << value << std::endl;
        result.emplace_back(stoi(value));
      }
      return result;
    }

    if constexpr (std::is_same_v<std::vector<std::string>, T>) {
      std::vector<std::string> result;
      for (auto value : args) {
        result.emplace_back(value);
      }
      return result;
    }
    // für morgen
    // wie bekomme ich den inneren Typ eines Vektors ?
  }
};
