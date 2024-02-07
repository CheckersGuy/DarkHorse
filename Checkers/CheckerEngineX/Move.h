//
// Created by Robin on 11.05.2017.
//

#ifndef CHECKERSTEST_MOVE_H
#define CHECKERSTEST_MOVE_H
#include "Bits.h"
#include "types.h"

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

  int get_move_encoding() const;

  static Move from_encoding(int encoding);

  // flips the move from white to black or vice versa
  Move flipped();

  friend std::ostream &operator<<(std::ostream &, Move other);
};

struct MoveEncoding {
  // How to encode empty moves ?
  MoveEncoding(Move move);
  MoveEncoding() = default;
  uint8_t from_index : 6;
  uint8_t direction : 2;
  Move get_move();
  void encode_move(Move move);
};

#endif // CHECKERSTEST_MOVE_H
