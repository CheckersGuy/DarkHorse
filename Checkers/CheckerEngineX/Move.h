//
// Created by Robin on 11.05.2017.
//

#ifndef CHECKERSTEST_MOVE_H
#define CHECKERSTEST_MOVE_H
#include "Bits.h"
#include <sstream>
struct Move {
  uint32_t from{0u};
  uint32_t to{0u};
  uint32_t captures{0u};

  bool is_capture() const;

  bool is_empty() const;

  bool operator==(const Move &other) const;

  bool operator!=(const Move &other) const;

  uint32_t get_from_index() const;

  uint32_t get_to_index() const;

  bool is_promotion(const uint32_t kings) const;

  bool is_pawn_move(const uint32_t kings) const;

  int num_captured() { return Bits::pop_count(captures); }

  int get_move_encoding() const;

  static Move from_encoding(int encoding);

  std::string get_move_as_string() const;

  // flips the move from white to black or vice versa
  Move flipped() {
    Move next;
    next.from = Bits::reverse32(from);
    next.to = Bits::reverse32(to);
    next.captures = Bits::reverse32(captures);
    return next;
  }

  friend std::ostream &operator<<(std::ostream &, Move other);
};

// 1 byte: from square (0..31) + direction (0..3) of a non-capture move; from_index 32
// = no move; from_index 40..49 = capture, stored as its index (0..39) in the move
// list of the position (generation order, see MoveListe / get_moves)
struct MoveEncoding {
  MoveEncoding(Move move);
  MoveEncoding() = default;
  uint8_t from_index : 6;
  uint8_t direction : 2;
  Move get_move() const; // empty for "no move" and for captures
  void encode_move(Move move);
  static MoveEncoding capture(int index) {
    MoveEncoding e;
    e.from_index = (uint8_t)(40 + index / 4);
    e.direction = (uint8_t)(index % 4);
    return e;
  }
  int capture_index() const { return from_index >= 40 ? (from_index - 40) * 4 + direction : -1; }
  bool is_none() const { return from_index == 32; }
  uint8_t raw() const { return (uint8_t)(from_index | (direction << 6)); }
};

#endif // CHECKERSTEST_MOVE_H
