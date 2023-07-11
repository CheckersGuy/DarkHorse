//
// Created by Robin on 09.05.2017.
//

#include "Board.h"
#include <cstdint>

Position &Board::get_position() { return pStack[pCounter]; }

size_t Board::history_length() const { return pStack.size(); }

Board::Board(const Board &other) {
  std::copy(other.pStack.begin(), other.pStack.end(), pStack.begin());
  this->pCounter = other.pCounter;
}

void Board::print_board() const {
  Position pos = pStack[pCounter];
  pos.print_position();
}

void Board::play_move(Move move) {
  if (move.is_capture() || move.is_pawn_move(get_position().K)) {
    this->last_non_rev = pCounter;
  }
  this->make_move(move);
}

void Board::make_move(Move move) {
  pStack[pCounter + 1] = pStack[pCounter];
  this->pCounter++;
  // Zobrist::update_zobrist_keys(get_position(), move);
  pStack[pCounter].make_move(move);
}

void Board::undo_move() { this->pCounter--; }

Board &Board::operator=(Position pos) {
  this->pCounter = 0;
  get_position().BP = pos.BP;
  get_position().WP = pos.WP;
  get_position().K = pos.K;
  get_position().color = pos.color;
  // get_position().key = Zobrist::generate_key(get_position());

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
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}
uint64_t Board::get_current_key() const {
  const Position p = pStack[pCounter];
  uint64_t first =
      static_cast<uint64_t>(p.BP) | (static_cast<uint64_t>(p.WP) << 32);
  uint64_t second =
      static_cast<uint64_t>(p.K) | (static_cast<uint64_t>(p.color) << 32);
  return hash_combine(hash(first), hash(second));
}

bool Board::is_repetition(int last_rev) const {
  auto end = std::max(last_rev - 1, 0);
  auto current = pStack[pCounter];
  int counter = 0;
  for (int i = pCounter; i >= end; i -= 2) {
    if (pStack[i] == current) {
      counter++;
    }
    if (counter >= 2) {
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
