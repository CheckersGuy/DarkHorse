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
void recurse(Board &board, std::unordered_set<Position> &hashset, int depth,
             Value min, Value max) {

  if (depth == 0) {
    Move bestMove;
    Board copy = board;
    auto it = hashset.find(board.get_position());
    // if we havent evaluated the position before, evaluate it now
    Value value = -INFINITE;
    if (it == hashset.end()) {
      value = searchValue(copy, bestMove, 0, 100000, false, std::cout);
      hashset.insert(board.get_position());
    }

    if (value >= min && value <= max) {
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

#define DB_PATH "E:\\kr_english_wld"
#define DTW_PATH "E:\\kr_english_dtw"

int main(int argl, const char **argc) {

#ifdef _WIN32
  tablebase.load_table_base(DB_PATH);
#endif
  /*
    Position test =
        Position::pos_from_fen("W:W32,30,28,27,26,25,19,15:B18,17,14,12,7,6,3,1");

    MoveL/diste liste;
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

  CmdParser parser;
  parser.parse(argl, argc);
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

  if (parser.has_option("book")) {
    std::string next_line;
    TT.resize_in_mb(2);
    while (std::getline(std::cin, next_line)) {
      // need to clear statistics all the time

      if (next_line == "terminate") {
        std::exit(-1);
      }
      const auto pos = Position::pos_from_fen(next_line);
      generate_book(10, pos, -115, 115);
      // sending a message, telling "master" to send us another position
      std::cout << "done" << std::endl;
    }
    return 0;
  }
  if (parser.has_option("generate")) {

    // const int adj_threshold = 350;
    // const float adj_percentage = 0.8f; // 80% of all games will be
    // adjudicated
    std::mt19937_64 generator(getSystemTime() ^ getpid());
    std::uniform_real_distribution<float> distrib(0, 1);

    std::string next_line;
    TT.resize_in_mb(128);
    std::vector<Position> rep_history;

    auto color_to_result = [](Color color) {
      return ((color == BLACK) ? BLACK_WON : WHITE_WON);
    };
    while (std::getline(std::cin, next_line)) {
      value_history.clear();
      if (next_line == "terminate") {
        std::exit(-1);
      }
      // bool do_adjudicate = (distrib(generator) < adj_percentage);
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

        auto value =
            searchValue(board, best, depth, time, max_nodes, false, std::cout);
        if (best.is_empty()) {
          // Just in case search could not finish
          result = UNKNOWN;
          break;
        }
        value_history.emplace_back(value);

        const auto kings = board.get_position().K;
        if (best.is_capture() || best.is_pawn_move(kings)) {
          value_history.clear();
        }

        board.play_move(best);
        auto count =
            std::count(rep_history.begin(), rep_history.end(),
                       (rep_history.empty()) ? Position{} : rep_history.back());
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
        // skipping terminal positions
        MoveListe check_liste;
        get_moves(rep_history[i], check_liste);
        if (check_liste.length() == 0)
          continue;

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
      TT.resize_in_mb(hash_size);
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
