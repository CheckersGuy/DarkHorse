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
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
INCBIN(mlh_net, "mlh3.quant");
INCBIN(network, "registry_128.quant");
INCBIN(policy, "policybigger6.quant");

INCBIN(mlh_perm, "mlh.perm");
INCBIN(net_perm, "evalpermutation.perm");
// INCBIN(policy_perm, "policy.perm");

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

struct SearchThread {

  std::thread local_thread;
  bool is_thinking = false; // if true, there is a search in progress
  Board search_board;
  bool stop_thread = false;

  void init() {

    while (!stop_thread) {
      // should accept some sort of search object
    }
  }
};

int main(int argl, const char **argc) {

  mlh_net.load_permutation_from_array(gmlh_permData, gmlh_permSize);
  // policy.load_permutation_from_array(gpolicy_permData, gpolicy_permSize);
  network.load_permutation_from_array(gnet_permData, gnet_permSize);

  mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
  network.load_from_array(gnetworkData, gnetworkSize);
  policy.load_from_array(gpolicyData, gpolicySize);

  CmdParser parser;
  parser.parse(argl, argc);
  Board board;

  std::vector<int> value_history;
  int time, depth, hash_size;
  size_t max_nodes = 18446744073709551615ull;
  std::string net_file;

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
    hash_size = 128;
  }

  if (parser.has_option("depth")) {
    depth = parser.as<int>("depth");
  } else {
    depth = parser.has_option("bench") ? 27 : MAX_PLY;
  }

  if (parser.has_option("search") || parser.has_option("bench"))

  {
    if (parser.has_option("position")) {
      auto pos_string = parser.as<std::string>("position");
      board.get_position() = Position::pos_from_fen(pos_string);
    } else {
      board.get_position() = Position::get_start_position();
    }
    board.get_position().print_position();

    TT.resize_in_mb(hash_size);
    Move best;
    if (parser.has_option("bench")) {
      searchValue(board, best, depth, time, max_nodes, false, std::cout);
    } else {
      searchValue(board, best, depth, time, max_nodes, true, std::cout);
    }

    return 0;
  }

  if (parser.has_option("eval-loop")) {
    TT.resize_in_mb(hash_size);
    std::string current;
    while (std::getline(std::cin, current)) {
      if (current == "terminate") {
        std::exit(-1);
      }

      TT.clear();
      const auto pos = Position::pos_from_fen(current);

      board = Board(pos);
      Move best;
      auto eval = searchValue(board, best, depth, time, max_nodes, false,
                              std::cout, false);

      std::cout << eval << std::endl;
    }
    return 0;
  }

  if (parser.has_option("book")) {
    std::string next_line;
    TT.resize_in_mb(2);
    while (std::getline(std::cin, next_line)) {
      // need to clear statistics all the time

      if (next_line == "terminate") {
        std::exit(-1);
      }
      const auto pos = Position::pos_from_fen(next_line);
      generate_book(8, pos, -100, 100);
      // sending a message, telling "master" to send us another position
      std::cout << "done" << std::endl;
    }
    return 0;
  }
  if (parser.has_option("generate")) {

    std::string next_line;
    TT.resize_in_mb(4);
    std::vector<Position> rep_history;
    std::vector<int> rep_values;

    auto color_to_result = [](Color color) {
      return ((color == BLACK) ? BLACK_WON : WHITE_WON);
    };
    while (std::getline(std::cin, next_line)) {
      if (next_line == "terminate") {
        std::exit(-1);
      }
      // bool do_adjudicate = (distrib(generator) < adj_percentage);
      TT.clear();
      const auto start_pos = Position::pos_from_fen(next_line);
      rep_history.clear();
      rep_values.clear();
      last_eval = -INFINITE;
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

        auto value = searchValue(board, best, depth, time, max_nodes, false,
                                 std::cout, false);
        if (best.is_empty()) {
          // Just in case search could not finish
          result = UNKNOWN;
          break;
        }
        const Position previous = board.get_position();
        board.play_move(best);
        auto count =
            std::count(rep_history.begin(), rep_history.end(),
                       (rep_history.empty()) ? Position{} : rep_history.back());
        if (count >= 3) {
          result = DRAW;
          break;
        }

        rep_history.emplace_back(previous);
        rep_values.emplace_back(value);
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
                  << res_to_string(result, rep_history[i].color) << "!"
                  << rep_values[i] << std::endl;
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
      TT.resize_in_mb(hash_size);
      std::cout << "init_ready"
                << "\n";
    } else if (current == "new_game") {
      last_eval = -INFINITE;
      TT.clear();
      TT.age_counter = 0u;
      std::string position;
      std::cin >> position;
      Position pos = Position::pos_from_fen(position);
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
    std::cout.flush();
  }
}
