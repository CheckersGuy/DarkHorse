//
// Created by robin on 7/26/18.
//

#ifndef TRAINING_MATCH_H
#define TRAINING_MATCH_H
#include "MGenerator.h"
#include "fcntl.h"
#include <MoveListe.h>
#include <Position.h>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

inline std::string getPositionString(Position pos) {
  std::string position;
  for (uint32_t i = 0; i < 32u; ++i) {
    uint32_t current = 1u << i;
    if ((current & (pos.BP & pos.K))) {
      position += "3";
    } else if ((current & (pos.WP & pos.K))) {
      position += "4";
    } else if ((current & pos.BP)) {
      position += "1";
    } else if ((current & pos.WP)) {
      position += "2";
    } else {
      position += "0";
    }
  }
  if (pos.get_color() == BLACK) {
    position += "B";
  } else {
    position += "W";
  }
  return position;
}

struct Engine {
  enum class State { Idle, Game_Ready, Init_Ready, Update };
  std::string engine_name;
  State state;
  const int &engineRead;
  const int &engineWrite;
  bool waiting_response = false;
  int time_move = 100;
  int hash_size = 21;

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
  int hash_size{21};
  int maxGames{1000000};
  int wins_one{0}, wins_two{0}, draws{0};
  int threads{1};
  std::vector<std::string> positions;
  std::string openingBook{"../Training/Positions/drawbook.book"};

public:
  explicit Match(const std::string &first, const std::string &second)
      : first(first), second(second) {
    std::ifstream stream(openingBook);

    if (!stream.good()) {
      std::cerr << "Could not load the openings" << std::endl;
      std::exit(-1);
    }
    std::string line;
    while (std::getline(stream, line)) {
      positions.emplace_back(line);
    }
    std::mt19937_64 generator(getSystemTime());
  };

  void setMaxGames(int games);

  int getMaxGames();

  void start();

  void set_arg1(std::string arg);

  void set_arg2(std::string arg);

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
