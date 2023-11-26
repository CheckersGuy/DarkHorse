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

  friend std::ostream &operator<<(std::ostream &, Move other);
};

#endif // CHECKERSTEST_MOVE_H
