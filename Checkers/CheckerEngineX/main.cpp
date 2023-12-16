#include "GameLogic.h"
#include "MGenerator.h"
#include "MovePicker.h"
#include "Perft.h"
#include "Selfplay.h"
#include "Transposition.h"
#include <cstdint>
#include <random>
#include <string>
#include <unistd.h>
#include <vector>
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

// game-generation

#include "CmdParser.h"
#include "Network.h"
#include "types.h"
int main(int argl, const char **argc) {

  CmdParser parser(argl, argc);
  parser.parse_command_line();
  Board board;
  Statistics::mPicker.init();

  int time, depth, hash_size;
  std::string net_file;

  if (parser.has_option("network")) {
    net_file = parser.as<std::string>("network");
  } else {
    // net_file = "int8test.quant";
    net_file = "relu.quant";
  }

  if (parser.has_option("time")) {
    time = parser.as<int>("time");
  } else {
    time = 100;
  }

  if (parser.has_option("hash_size")) {
    hash_size = parser.as<int>("hash_size");
  } else {
    hash_size = 22;
  }

  network.load_bucket(net_file);
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

    TT.resize(hash_size);
    Move best;
    if (parser.has_option("bench")) {
      searchValue(board, best, depth, time, false, std::cout);
    } else {
      searchValue(board, best, depth, time, true, std::cout);
    }
    std::cout << "NodeCounter: " << nodeCounter << std::endl;
    return 0;
  }

  if (parser.has_option("selfplay")) {
    Selfplay selfplay;
    int hash_size;
    if (parser.has_option("hash_size")) {
      hash_size = parser.as<int>("hash_size");
    } else {
      hash_size = 22;
    }
    TT.resize(hash_size);

    int time;
    if (parser.has_option("time")) {
      time = parser.as<int>("time");
    } else {
      time = 10;
    }

    selfplay.start_loop();

    return 0;
  }

  //
  if (parser.has_option("eval_loop")) {
    std::string next_line;
    TT.resize(16);
    while (std::getline(std::cin, next_line)) {
      // need to clear statistics all the time

      if (next_line == "terminate") {
        std::exit(-1);
      }
      Statistics::mPicker.clear_scores();
      TT.clear();

      board.get_position() = Position::pos_from_fen(next_line);
      Move bestMove;
      auto s_val = searchValue(board, bestMove, MAX_PLY, 10, false, std::cout);
      std::cout << next_line << "!" << s_val << std::endl;
    }
  }

  if (parser.has_option("generate")) {

    Statistics::mPicker.init();
    std::string next_line;
    TT.resize(20);
    std::vector<Position> rep_history;
    Statistics::mPicker.clear_scores();
    while (std::getline(std::cin, next_line)) {
      // nextline should be a fen_string
      if (next_line == "terminate") {
        std::exit(-1);
      }
      TT.clear();
      Statistics::mPicker.clear_scores();
      const auto start_pos = Position::pos_from_fen(next_line);
      rep_history.clear();

      board = Board(start_pos);
      Result result = UNKNOWN;
      for (auto i = 0; i < 600; ++i) {

        rep_history.emplace_back(board.get_position());
        Move best;
        MoveListe liste;
        get_moves(board.get_position(), liste);
        if (liste.length() == 0) {
          result = ((board.get_mover() == BLACK) ? WHITE_WON : BLACK_WON);
          break;
        }

        searchValue(board, best, MAX_PLY, time, false, std::cout);
        board.play_move(best);

        const auto last_position = rep_history.back();
        auto count =
            std::count(rep_history.begin(), rep_history.end(), last_position);
        if (count >= 3) {
          result = DRAW;
          break;
        }
      }
      // sending all the the results back
      std::cout << "BEGIN" << std::endl;
      for (auto position : rep_history) {
        std::string result_string = "UNKNOWN";

        if ((result == BLACK_WON && position.color == BLACK) ||
            (result == WHITE_WON && position.color == WHITE)) {
          result_string = "WON";
        } else if ((result == BLACK_WON && position.color != BLACK) ||
                   (result == WHITE_WON && position.color != WHITE)) {
          result_string = "LOSS";
        } else if (result == DRAW) {
          result_string = "DRAW";
        }

        if (position.get_color() == BLACK) {
          position = position.get_color_flip();
        }

        std::cout << position.get_fen_string() << "!" << result_string
                  << std::endl;
      }
      std::cout << "END" << std::endl;
    }
  }

  std::string current;
  while (std::cin >> current) {
    if (current == "init") {
      TT.age_counter = 0u;
      Statistics::mPicker.clear_scores();
      std::string hash_string;
      std::cin >> hash_string;
      const int hash_size = std::stoi(hash_string);
      TT.resize(hash_size);
      std::cout << "init_ready"
                << "\n";
    } else if (current == "new_game") {
      TT.clear();
      Statistics::mPicker.clear_scores();
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
      Statistics::mPicker.clear_scores();
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
