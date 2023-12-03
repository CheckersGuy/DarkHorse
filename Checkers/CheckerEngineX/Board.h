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

extern std::ofstream debug;

class Board {

public:
  std::array<Position, MAX_PLY> pStack;
  std::array<Position, 400> rep_history;
  std::optional<Color> color_us; // will be set whenever we start a search
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

  bool is_repetition(int last_rev) const;

  Color get_mover() const;

  Board &operator=(Position pos);

  size_t history_length() const;

  Position history_at(size_t idx) const;

  void play_move(Move move);

  Board &operator=(const Board &other);
};

#endif // CHECKERSTEST_BOARD_H
