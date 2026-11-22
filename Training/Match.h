//
// Created by robin on 7/26/18.
//

#ifndef TRAINING_MATCH_H
#define TRAINING_MATCH_H
#include "MGenerator.h"
#include "Sprt.h"
#include "fcntl.h"
#include <MoveListe.h>
#include <Position.h>
#include <chrono>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

struct Engine {
  enum class State { Idle, Game_Ready, Init_Ready, Update };
  std::string engine_name;
  State state;
  const int &engineRead;
  const int &engineWrite;
  bool waiting_response = false;
  int time_move = 100;
  int hash_size = 21;

  // CheckerBoard-style time bank for fixed time per move (engine matches):
  // before each move the bank is credited with time_move and the engine gets
  // maxtime = cb_maxtime(bank, time_move); afterwards the time actually used
  // is debited. Reset at every new game.
  bool use_bank = false;
  double bank_ms = 0.0;
  int sent_maxtime = 0;
  std::chrono::steady_clock::time_point search_start;
  // statistics over the whole match
  uint64_t searches = 0;
  double used_ms_sum = 0.0, maxtime_ms_sum = 0.0;

  void initEngine();

  void writeMessage(const std::string &msg);

  void new_move(Move move);

  void newGame(const Position &pos);

  void update();

  Move search();

  std::string readPipe();

  void setHashSize(int hash);

  void setTime(int time);
};

struct Interface {

  std::array<Engine, 2> engines;
  Position start_pos;
  Position pos;
  bool first_game = true;
  int first_mover = 0;
  std::vector<Position> history;
  size_t pos_counter{0};
  // engine 1's result of the first game of the current pair, in half points
  // (0 loss, 1 draw, 2 win); -1 while no first game is pending
  int pending_result{-1};

  void process();

  bool is_n_fold(int n);

  bool is_legal_move(Move move);

  bool is_terminal_state();

  void reset_engines();

  void terminate_engines();
};

class Match {

private:
  size_t opening_counter{0};
  const std::string &first;
  const std::string &second;
  std::string arg1, arg2;
  int time{100};
  std::pair<int, int> times;
  int hash_size{128};
  int maxGames{1000000};
  int wins_one{0}, wins_two{0}, draws{0};
  int threads{1};
  bool time_bank{false};
  bool use_book_seed{false};
  uint64_t book_seed{0};
  std::vector<std::string> positions;
  std::string openingBook{"../Training/Positions/drawbook.book"};

  // pair statistics and SPRT
  Sprt::Pentanomial penta;
  bool sprt_enabled{false};
  double elo0{0.0}, elo1{5.0}, alpha{0.05}, beta{0.05};

  void load_book();
  void print_status(int games);

public:
  explicit Match(const std::string &first, const std::string &second)
      : first(first), second(second) {
    load_book();
  };

  void set_sprt(double elo0, double elo1, double alpha, double beta);

  void set_time_bank(bool enable) { time_bank = enable; }

  // random (seeded) start offset in the book for every parallel game
  void set_book_seed(uint64_t seed) {
    use_book_seed = true;
    book_seed = seed;
  }

  bool is_sprt() const { return sprt_enabled; }

  void setMaxGames(int games);

  int getMaxGames();

  void start();

  void set_arg1(std::string arg);

  void set_arg2(std::string arg);

  const std::string &get_arg1() const { return arg1; }

  const std::string &get_arg2() const { return arg2; }

  void setHashSize(int hash);

  void setTime(int time);

  void setNumThreads(int threads);

  int getNumThreads();

  const std::string &getOpeningBook();

  void setOpeningBook(std::string book);

  std::string get_start_pos(size_t &pos_counter);

  void set_time(int time_one, int time_two);
};

#endif // TRAINING_MATCH_H
