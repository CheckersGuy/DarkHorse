#include "Bits.h"
#include "CmdParser.h"
#include "GameLogic.h"
#include "MGenerator.h"
#include "Network.h"
#include "Perft.h"
#include "Transposition.h"
#include "incbin.h"
#include "types.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <hash_set>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
INCBIN(mlh_net, "mlh3.quant");
INCBIN(network, "moesuper.quant");
INCBIN(policy, "policybigger2.quant");
inline Position posFromString(const std::string &pos) {
  Position result;
  for (uint32_t i = 0; i < 32u; ++i) {
    uint32_t current = 1u << i;
    if (pos[i] == '1') {
      result.BP |= current;
    } else if (pos[i] == '2') {
      result.WP |= current;
    } else if (pos[i] == '3') {
      result.K |= current;
      result.BP |= current;
    } else if (pos[i] == '4') {
      result.K |= current;
      result.WP |= current;
    }
  }
  if (pos[32] == 'B') {
    result.color = BLACK;
  } else {
    result.color = WHITE;
  }
  return result;
}
std::vector<std::string> split(const std::string &s, char delim) {
  std::vector<std::string> result;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    result.push_back(item);
  }

  return result;
}

void recurse(Board &board, std::unordered_set<Position> &hashset, int depth,
             Value min, Value max) {

  if (depth == 0 || board.get_position().piece_count() <= 16) {
    Move bestMove;
    TT.clear();
    Board copy = board;
    auto it = hashset.find(board.get_position());
    // if we havent evaluated the position before, evaluate it now
    Value value = -INFINITE;
    if (it == hashset.end()) {
      value = searchValue(copy, bestMove, 0, 100000, false, std::cout);
      hashset.insert(board.get_position());
    }

    if (value >= min && value <= max && !board.get_position().has_jumps()) {
      std::cout << board.get_position().get_fen_string() << std::endl;
    }
    return;
  }

  MoveListe liste;
  get_moves(board.get_position(), liste);

  for (auto i = 0; i < liste.length(); ++i) {
    const Move move = liste[i];
    board.make_move(move);
    recurse(board, hashset, depth - 1, min, max);
    board.undo_move();
  }

  return;
}

void generate_book(int depth, Position pos, Value min_value, Value max_value) {
  std::unordered_set<Position> hashset;
  Board board(pos);
  recurse(board, hashset, depth, min_value, max_value);
}

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

struct NewCmdParser {

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

#define DB_PATH "E:\\kr_english_wld"
#define DTW_PATH "E:\\kr_english_dtw"

int main(int argl, const char **argc) {

  NewCmdParser parsertest;
  parsertest.parse(argl, argc);
  std::cout << parsertest.as<std::vector<int>>("time").size() << std::endl;
  /*
    for (auto [key, value] : parsertest.options) {
      std::cout << "Key: " << key << " and Values: ";
      std::copy(value.begin(), value.end(),
                std::ostream_iterator<std::string>(std::cout, " "));
      std::cout << std::endl;
    }
  */
  return 0;

#ifdef _WIN32
  tablebase.load_table_base(DB_PATH);
#endif
  /*
    Position test =
        Position::pos_from_fen("W:W32,30,28,27,26,25,19,15:B18,17,14,12,7,6,3,1");

    MoveListe liste;
    get_moves(test, liste);

    for (auto m : liste) {
      std::cout << m.get_move_encoding() << std::endl;
    }
    return 0;
  */
  mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
  network.load_from_array(gnetworkData, gnetworkSize);
  policy.load_from_array(gpolicyData, gpolicySize);
  /*
    network.print_layers();
    policy.print_layers();
    mlh_net.print_layers();

    Position pos = Position::pos_from_fen("B:WK5,K26:B4,3,1");
    pos.print_position();

    std::cout << network.evaluate(pos, 0, 0);
    return 0;
    */
  CmdParser parser(argl, argc);
  parser.parse_command_line();
  Board board;

  std::vector<int> value_history;
  int time, depth, hash_size;
  size_t max_nodes = 18446744073709551615ull;
  std::string net_file;

  if (parser.has_option("network")) {
    net_file = parser.as<std::string>("network");
    network.load_bucket(net_file);
  }

  if (parser.has_option("time")) {
    time = parser.as<int>("time");
  } else {
    time = 100;
  }
  if (parser.has_option("nodes")) {
    max_nodes = parser.as<int>("nodes");
  } else {
    max_nodes = 18446744073709551615ull;
  }

  if (parser.has_option("hash_size")) {
    hash_size = parser.as<int>("hash_size");
  } else {
    hash_size = 21;
  }

  if (parser.has_option("search") || parser.has_option("bench"))

  {
    if (parser.has_option("depth")) {
      depth = parser.as<int>("depth");
    } else {
      depth = parser.has_option("bench") ? 27 : MAX_PLY;
    }

    if (parser.has_option("position")) {
      auto pos_string = parser.as<std::string>("position");
      board.get_position() = Position::pos_from_fen(pos_string);
    } else {
      board.get_position() = Position::get_start_position();
    }
    board.get_position().print_position();

    TT.resize(hash_size);
    Move best;
    if (parser.has_option("bench")) {
      searchValue(board, best, depth, time, max_nodes, false, std::cout);
    } else {
      searchValue(board, best, depth, time, max_nodes, true, std::cout);
    }

    return 0;
  }

  if (parser.has_option("book")) {
    std::string next_line;
    TT.resize(2);
    while (std::getline(std::cin, next_line)) {
      // need to clear statistics all the time

      if (next_line == "terminate") {
        std::exit(-1);
      }
      const auto pos = Position::pos_from_fen(next_line);
      generate_book(8, pos, -30, 30);
      // sending a message, telling "master" to send us another position
      std::cout << "done" << std::endl;
    }
    return 0;
  }
  if (parser.has_option("generate")) {

    const int adj_threshold = 350;
    const float adj_percentage = 0.8f; // 80% of all games will be adjudicated
    std::mt19937_64 generator(getSystemTime() ^ getpid());
    std::uniform_real_distribution<float> distrib(0, 1);

    std::string next_line;
    TT.resize(18);
    std::vector<Position> rep_history;

    auto color_to_result = [](Color color) {
      return ((color == BLACK) ? BLACK_WON : WHITE_WON);
    };
    while (std::getline(std::cin, next_line)) {
      value_history.clear();
      if (next_line == "terminate") {
        std::exit(-1);
      }
      bool do_adjudicate = (distrib(generator) < adj_percentage);
      TT.clear();
      const auto start_pos = Position::pos_from_fen(next_line);
      rep_history.clear();

      board = start_pos;
      Result result = UNKNOWN;
      for (auto i = 0; i < 600; ++i) {
        Move best;
        MoveListe liste;
        get_moves(board.get_position(), liste);
        if (liste.length() == 0) {
          // we dont want those positions in our history
          // since they are not evaluated by the network anyways
          result = ((board.get_mover() == BLACK) ? WHITE_WON : BLACK_WON);
          break;
        }

        auto value = searchValue(board, best, MAX_PLY, time, max_nodes, false,
                                 std::cout);
        if (best.is_empty()) {
          // Just in case search could not finish
          result = UNKNOWN;
          break;
        }
        if (std::abs(value) >= adj_threshold && do_adjudicate && (i >= 15) &&
            (board.get_position().piece_count() <= 10)) {
          if (value > 0) {
            result = ((board.get_mover() == BLACK) ? BLACK_WON : WHITE_WON);
          } else {
            result = ((board.get_mover() == BLACK) ? WHITE_WON : BLACK_WON);
          }
          break;
        }
        // computing the exponential moving average;
        value_history.emplace_back(value);

        if (value_history.size() >= 40) {
          double average = 0;
          for (auto i = 0; i < 40; i++) {
            average += std::abs(value_history[value_history.size() - 1 - i]);
          }
          average /= 40.0;
          if (average <= 3 && (board.get_position().piece_count() <= 10)) {
            result = DRAW;
            break;
          }
        }

        const auto kings = board.get_position().K;
        if (best.is_capture() || best.is_pawn_move(kings)) {
          value_history.clear();
        }

        board.play_move(best);

        const auto last_position =
            (rep_history.size() > 0) ? rep_history.back() : Position{};
        auto count =
            std::count(rep_history.begin(), rep_history.end(), last_position);
        if (count >= 3) {
          result = DRAW;
          break;
        }

        rep_history.emplace_back(board.get_position());
      }

      auto res_to_string = [](Result result, Color color) {
        if ((result == BLACK_WON && color == BLACK) ||
            (result == WHITE_WON && color == WHITE)) {
          return "WON";
        } else if ((result == BLACK_WON && color != BLACK) ||
                   (result == WHITE_WON && color != WHITE)) {
          return "LOSS";
        } else if (result == DRAW) {
          return "DRAW";
        } else {
          return "UNKNOWN";
        }
      };

      // sending all the the results back in reverse order
      std::cout << "BEGIN" << std::endl;
      for (int i = rep_history.size() - 1; i >= 0; --i) {
        std::cout << rep_history[i].WP << "!" << rep_history[i].BP << "!"
                  << rep_history[i].K << "!" << (int)rep_history[i].color << "!"
                  << res_to_string(result, rep_history[i].color) << std::endl;
      }
      std::cout << "END" << std::endl;
    }
  }

  std::string current;
  while (std::cin >> current) {
    if (current == "init") {
      TT.age_counter = 0u;
      std::string hash_string;
      std::cin >> hash_string;
      const int hash_size = std::stoi(hash_string);
      TT.resize(hash_size);
      std::cout << "init_ready"
                << "\n";
    } else if (current == "new_game") {
      TT.clear();
      TT.age_counter = 0u;
      std::string position;
      std::cin >> position;
      Position pos = posFromString(position);
      board = Board(pos);
      std::cout << "game_ready"
                << "\n";
    } else if (current == "new_move") {
      // opponent made a move and we need to update the board
      Move move;
      std::vector<uint32_t> squares;
      std::string line;
      std::cin >> line;
      while (!line.empty()) {
        if (line == "end_move")
          break;
        squares.emplace_back(std::stoi(line));
        std::cin >> line;
      }
      move.from = 1u << squares[0];
      move.to = 1u << squares[1];
      for (auto i = 2; i < squares.size(); ++i) {
        move.captures |= 1u << squares[i];
      }

      board.play_move(move);
      std::cout << "update_ready"
                << "\n";
    } else if (current == "search") {
      std::string time_string;
      std::cin >> time_string;
      Move bestMove;
      searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false,
                  std::cout);
      std::cout << "new_move"
                << "\n";
      std::cout << std::to_string(bestMove.get_from_index()) << "\n";
      std::cout << std::to_string(bestMove.get_to_index()) << "\n";
      uint32_t captures = bestMove.captures;
      while (captures) {
        std::cout << std::to_string(Bits::bitscan_foward(captures)) << "\n";
        captures &= captures - 1u;
      }
      std::cout << "end_move"
                << "\n";

      board.play_move(bestMove);
      // adding the move to the repetition history for our side

    } else if (current == "terminate") {
      // terminating the program
      break;
    }
  }
}
