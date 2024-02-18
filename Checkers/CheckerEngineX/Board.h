//
// Created by Robin on 09.05.2017.
//

#ifndef CHECKERSTEST_BOARD_H
#define CHECKERSTEST_BOARD_H

#include "Move.h"
#include "Network.h"
#include "Position.h"
#include "types.h"
#include <string>

// extern std::ofstream debug;

class Board {

public:
  std::array<Position, MAX_PLY> pStack;
  std::array<Position, 800> rep_history;
  std::array<int16_t, MAX_PLY> last_rev;
  int rep_size = 0;

public:
  int pCounter = 0;

  Board() = default;

  Board(Position pos);

  void print_board() const;

  void reset(Position pos);

  void make_move(Move move);

  void undo_move();

  bool is_silent_position();

  Position &get_position();

  Position previous() const;

  uint64_t get_current_key() const;

  bool is_repetition() const;

  Color get_mover() const;

  Board &operator=(Position pos);

  size_t history_length() const;

  Position history_at(size_t idx) const;

  void play_move(Move move);
};

#endif // CHECKERSTEST_BOARD_H
