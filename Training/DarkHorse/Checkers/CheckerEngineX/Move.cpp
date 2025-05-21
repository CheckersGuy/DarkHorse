//
// Created by robin on 10.10.19.
//
#include "Move.h"
#include "Position.h"

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

int Move::get_move_encoding() const {
  int dir = 0;

  if ((((from & MASK_L3) << 3) == to) || (((from & MASK_L5) << 5) == to)) {
    dir = 0;

  } else if (((from) << 4) == to) {
    dir = 1;

  } else if (((from) >> 4) == to) {
    dir = 2;

  } else if ((((from & MASK_R3) >> 3) == to) ||
             (((from & MASK_R5) >> 5) == to)) {
    dir = 3;
  };

  return 4 * get_from_index() + dir;
}

void MoveEncoding::encode_move(Move move) {
  from_index = (!move.is_empty()) ? move.get_from_index() : 32;
  if (move.is_empty())
    return;
  uint8_t dir;
  if ((((move.from & MASK_L3) << 3) == move.to) ||
      (((move.from & MASK_L5) << 5) == move.to)) {
    dir = 0;

  } else if (((move.from) << 4) == move.to) {
    dir = 1;

  } else if (((move.from) >> 4) == move.to) {
    dir = 2;

  } else if ((((move.from & MASK_R3) >> 3) == move.to) ||
             (((move.from & MASK_R5) >> 5) == move.to)) {
    dir = 3;
  };
  direction = dir;
}

MoveEncoding::MoveEncoding(Move m) { encode_move(m); }

Move MoveEncoding::get_move() {
  if (from_index == 32) {
    return Move{};
  }
  Move move;
  uint32_t from = 1u << from_index;
  uint32_t to;
  if (direction == 3) {
    to = ((from & MASK_R3) >> 3) | ((from & MASK_R5) >> 5);
  } else if (direction == 2) {

    to = from >> 4;
  } else if (direction == 1) {
    to = from << 4;
  } else if (direction == 0) {
    to = ((from & MASK_L3) << 3) | ((from & MASK_L5) << 5);
  };
  move.from = from;
  move.to = to;
  return move;
}

Move Move::from_encoding(int encoding) {
  uint32_t orig = encoding / 4;
  uint32_t dir = encoding % 4;
  MoveEncoding encode;
  encode.from_index = orig;
  encode.direction = dir;
  return encode.get_move();
}

Move Move::flipped() {
  Move next;
  next.from = getMirrored(from);
  next.to = getMirrored(to);
  next.captures = getMirrored(captures);
  return next;
}
