#include "Bits.h"
#include "CmdParser.h"
#include "GameLogic.h"
#include "MGenerator.h"
#include "MovePicker.h"
#include "Network.h"
#include "Perft.h"
#include "Selfplay.h"
#include "Transposition.h"
#include "types.h"
#include <cstdint>
#include <cstdlib>
#include <hash_set>
#include <random>
#include <string>
#include <unistd.h>
#include <unordered_set>
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

  if (depth == 0) {
    Move bestMove;
    Statistics::mPicker.clear_scores();
    TT.clear();
    Board copy = board;
    auto it = hashset.find(board.get_position());
    // if we havent evaluated the position before, evaluate it now
    Value value = -INFINITE;
    if (it == hashset.end()) {
      value = searchValue(copy, bestMove, 1, 100000, false, std::cout);
      hashset.insert(board.get_position());
    }

    if (value >= min && value <= max && !board.get_position().has_jumps()) {
      std::cout << board.get_position().get_fen_string() << std::endl;
      // std::cout << "After evaluation: " << value << std::endl << std::endl;
      // board.get_position().print_position();
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

void print_msgs(char *msg) { printf("%s", msg); }

#ifdef USEDB

Result get_tb_result(Position pos, int max_pieces, EGDB_DRIVER *handle) {
  if (pos.has_jumps() || Bits::pop_count(pos.BP | pos.WP) > max_pieces)
    return UNKNOWN;

  EGDB_NORMAL_BITBOARD board;
  board.white = pos.WP;
  board.black = pos.BP;
  board.king = pos.K;

  EGDB_BITBOARD normal;
  normal.normal = board;
  auto val = handle->lookup(
      handle, &normal, ((pos.color == BLACK) ? EGDB_BLACK : EGDB_WHITE), 0);

  if (val == EGDB_UNKNOWN)
    return UNKNOWN;

  if (val == EGDB_WIN)
    return (pos.color == BLACK) ? BLACK_WON : WHITE_WON;

  if (val == EGDB_LOSS)
    return (pos.color == BLACK) ? WHITE_WON : BLACK_WON;

  if (val == EGDB_DRAW)
    return DRAW;

  return UNKNOWN;
}

#endif

// game-generation

// adding endgame table_bases for testing purposes

#define DB_PATH "E:\\kr_english_wld"

int main(int argl, const char **argc) {

#ifdef USEDB
  int i, status, max_pieces, nerrors;
  EGDB_TYPE egdb_type;
  EGDB_DRIVER *handle;

  /* Check that db files are present, get db type and size. */
  status = egdb_identify(DB_PATH, &egdb_type, &max_pieces);
  std::cout << "MAX_PIECES: " << max_pieces << std::endl;

  if (status) {
    printf("No database found at %s\n", DB_PATH);
    return (1);
  }
  printf("Database type %d found with max pieces %d\n", egdb_type, max_pieces);

  /* Open database for probing. */
  handle = egdb_open(EGDB_NORMAL, max_pieces, 1000, DB_PATH, print_msgs);
  if (!handle) {
    printf("Error returned from egdb_open()\n");
    return (1);
  }
#endif

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
    net_file = "newopen11.quant";
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

#ifdef USEDB
  if (parser.has_option("endgame")) {
    std::cout << "Testing" << std::endl;
    Position pos = Position::pos_from_fen("W:WK6:B4,3");

    pos.print_position();
    auto result = get_tb_result(pos, max_pieces, handle);
    auto result_string = (result == WHITE_WON)   ? "WHITE_WON"
                         : (result == BLACK_WON) ? "BLACK_WON"
                         : (result == DRAW)      ? "DRAW"
                                                 : "UNKNOWN";
    std::cout << result_string << std::endl;
    return 0;
  }
#endif

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
      searchValue(board, best, depth, time, false, std::cout);
    } else {
      searchValue(board, best, depth, time, true, std::cout);
    }
    return 0;
  }

  if (parser.has_option("book")) {
    std::string next_line;
    TT.resize(2);
    Statistics::mPicker.init();
    while (std::getline(std::cin, next_line)) {
      // need to clear statistics all the time

      if (next_line == "terminate") {
        std::exit(-1);
      }
      const auto pos = Position::pos_from_fen(next_line);
      generate_book(9, pos, -150, 150);
      // sending a message, telling "master" to send us another position
      std::cout << "done" << std::endl;
    }
    return 0;
  }

  if (parser.has_option("generate")) {
    int child_id = -1;
    Statistics::mPicker.init();
    std::string next_line;
    TT.resize(19);
    std::vector<Position> rep_history;
    Statistics::mPicker.clear_scores();
    while (std::getline(std::cin, next_line)) {
      // nextline should be a fen_string
      if (next_line == "terminate") {
#ifdef USEDB
        handle->close(handle);
#endif
        std::exit(-1);
      }
      TT.clear();
      Statistics::mPicker.clear_scores();
      const auto start_pos = Position::pos_from_fen(next_line);
      rep_history.clear();

      board = Board(start_pos);
      Result result = UNKNOWN;
      for (auto i = 0; i < 600; ++i) {

        Move best;
        MoveListe liste;
        get_moves(board.get_position(), liste);
        if (liste.length() == 0) {
          result = ((board.get_mover() == BLACK) ? WHITE_WON : BLACK_WON);
          break;
        }

        rep_history.emplace_back(board.get_position());
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
        auto position = rep_history[i];
        std::string result_string = "";
#ifdef USEDB
        auto local_result = get_tb_result(position, max_pieces, handle);
        if (local_result != UNKNOWN) {
          result = local_result;
          result_string = "TB_";
        }
#endif
        result_string.append(res_to_string(result, position.color));
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
