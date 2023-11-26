//
// Created by robin on 10.10.19.
//
#include "Move.h"

bool Move::is_pawn_move(const uint32_t kings) const {
  return (from & kings) == 0;
}

bool Move::is_capture() const { return captures != 0u; }

bool Move::is_empty() const {
  return (from == 0u) && (to == 0u) && (captures == 0u);
}

uint32_t Move::get_from_index() const { return Bits::bitscan_foward(from); }

uint32_t Move::get_to_index() const { return Bits::bitscan_foward(to); }

bool Move::operator==(const Move &other) const {
  return (to == other.to) && (from == other.from) &&
         (captures == other.captures);
}

bool Move::operator!=(const Move &other) const { return !((*this) == other); }

bool Move::is_promotion(const uint32_t kings) const {
  constexpr uint32_t promo_squares = PROMO_SQUARES_BLACK | PROMO_SQUARES_WHITE;
  return (((from & kings) == 0u) && ((to & promo_squares) != 0u));
}

std::ostream &operator<<(std::ostream &stream, Move other) {
  stream << "From: " << other.get_from_index()
         << " To: " << other.get_to_index();
  ;
  return stream;
}
