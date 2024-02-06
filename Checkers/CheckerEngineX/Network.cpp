//
// Created by root on 18.04.21.
//

#include "Network.h"
#include "Bits.h"
#include "GameLogic.h"
#include "types.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

Value tempo_white(Position pos) {
  Value score = 0;
  for (auto i = 0; i < 7; ++i) {
    score += (i + 1) * Bits::pop_count(MASK_ROWS[i] & (pos.WP & (~pos.K)));
  }
  return score;
}
Value tempo_black(Position pos) {
  Value score = 0;
  for (auto i = 7; i >= 1; --i) {
    score += (8 - i) * Bits::pop_count(MASK_ROWS[i] & (pos.BP & (~pos.K)));
  }
  return score;
}

Value win_eval(TB_RESULT result, Value score, Position pos) {
  // helper to encourage finishing the game
  auto BK = pos.BP & pos.K;
  auto WK = pos.WP & pos.K;
  auto total_pieces = pos.piece_count();
  auto eval = 100 * (Bits::pop_count(pos.WP & (~pos.K)) -
                     Bits::pop_count(pos.BP & (~pos.K)));
  eval +=
      140 * (Bits::pop_count(pos.WP & pos.K) - Bits::pop_count(pos.BP & pos.K));

  eval += (Bits::pop_count(pos.WP) -
           Bits::pop_count(pos.BP) * (240 - total_pieces * 20));

  eval += (Bits::pop_count(WK & CENTER) - Bits::pop_count(BK & CENTER)) * 8;
  // penalty for having kings in a single corner
  eval += (Bits::pop_count(WK & SINGLE_CORNER) -
           Bits::pop_count(BK & SINGLE_CORNER)) *
          -25;

  if (result == TB_RESULT::WIN) {
    if (pos.color == WHITE) {
      eval += 5 * tempo_white(pos);
    } else {
      eval -= 5 * tempo_black(pos);
    }
  }

  return score + eval * pos.color;
}
