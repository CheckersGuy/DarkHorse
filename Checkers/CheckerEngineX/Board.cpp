//
//
// Created by Robin on 09.05.2017.
//

#include "Board.h"
#include "types.h"
#include <cstdint>
#include <optional>
#include <unistd.h>
// std::ofstream debug("debug.txt");
Position &Board::get_position() { return pStack[pCounter]; }

size_t Board::history_length() const { return pStack.size(); }

Board::Board(Position pos) {
  this->pCounter = 0;
  get_position() = pos;
  rep_size = 0;
  // color_us will be set in search;
}

void Board::reset(Position start_pos) {
  pCounter = 0;
  pStack[pCounter] = start_pos;
}

void Board::print_board() const {
  Position pos = pStack[pCounter];
  pos.print_position();
}

void Board::play_move(Move move) {
  Position copy = get_position();
  copy.make_move(move);
  rep_history[rep_size++] = copy;
  pCounter = 0;
  pStack[pCounter] = copy;
  last_rev[pCounter] = 0;
}

void Board::make_move(Move move) {
  if (move.is_capture() || move.is_pawn_move(get_position().K)) {
    last_rev[pCounter + 1] = pCounter;
  } else {
    last_rev[pCounter + 1] = last_rev[pCounter];
  }
  pStack[pCounter + 1] = pStack[pCounter];
  this->pCounter++;

  pStack[pCounter].make_move(move);
}

void Board::undo_move() { this->pCounter--; }

Board &Board::operator=(Position pos) {
  this->pCounter = 0;
  get_position() = pos;
  rep_size = 0;

  return *this;
}

Color Board::get_mover() const { return pStack[pCounter].get_color(); }

bool Board::is_silent_position() {
  return (get_position().get_jumpers<WHITE>() == 0u &&
          get_position().get_jumpers<BLACK>() == 0u);
}

uint64_t hash_combine(uint64_t lhs, uint64_t rhs) {
  lhs ^= rhs + 0x9e3779b9 + (lhs << 6) + (lhs >> 2);
  return lhs;
}

uint64_t hash(uint64_t x) {
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 32;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 32;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 32;
  return x;
}
uint64_t Board::get_current_key() const {
  const Position p = pStack[pCounter];
  uint64_t first = static_cast<uint64_t>(p.BP & (~p.K)) |
                   (static_cast<uint64_t>(p.WP & (~p.K)) << 32);
  uint64_t second = static_cast<uint64_t>(p.BP & (p.K)) |
                    (static_cast<uint64_t>(p.WP & (p.K)) << 32);

  auto comb_hash = hash_combine(hash(first), hash(second));
  if (get_mover() == BLACK) {
    comb_hash ^= BLACK_RANDOM ^ getpid();
  }

  return comb_hash;
}

bool Board::is_repetition() const {
  const auto end = std::max(last_rev[pCounter] - 1, 0);
  const auto current = pStack[pCounter];
  for (int i = pCounter - 2; i >= end; i -= 2) {
    if (pStack[i] == current) {
      return true;
    }
  }

  return false;
}

Position Board::previous() const {
  if (pCounter > 0) {
    return pStack[pCounter - 1];
  }
  return Position{};
}

Position Board::history_at(size_t idx) const { return pStack[idx]; }
