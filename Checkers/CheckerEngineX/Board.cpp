//
//
// Created by Robin on 09.05.2017.
//

#include "Board.h"
#include "types.h"
#include <cstdint>
#include <optional>
Position &Board::get_position() { return pStack[pCounter]; }

size_t Board::history_length() const { return pStack.size(); }

Board::Board(const Board &other) {
  std::copy(other.pStack.begin(), other.pStack.end(), pStack.begin());
  this->pCounter = other.pCounter;
  this->rep_size = other.rep_size;
  std::copy(other.rep_history.begin(), other.rep_history.end(),
            rep_history.begin());
}

// constructs a new board with the given starting position

Board::Board(Position pos) {
  this->pCounter = 0;
  get_position().BP = pos.BP;
  get_position().WP = pos.WP;
  get_position().K = pos.K;
  get_position().color = pos.color;
  rep_size = 0;
  // color_us will be set in search;
}

Board &Board::operator=(const Board &other) {
  std::copy(other.pStack.begin(), other.pStack.end(), pStack.begin());
  this->pCounter = other.pCounter;
  this->rep_size = other.rep_size;
  std::copy(other.rep_history.begin(), other.rep_history.end(),
            rep_history.begin());
  return *this;
}

void Board::print_board() const {
  Position pos = pStack[pCounter];
  pos.print_position();
}

void Board::play_move(Move move) {
  Position copy = get_position();
  const Position temp = copy;

  const bool is_not_rev = move.is_capture() || move.is_pawn_move(copy.K);
  copy.make_move(move);
  if (is_not_rev && copy.color == color_us) {
    rep_size = 0;
  }
  if (temp.color == color_us) {
    rep_history[rep_size++] = temp;
  }
  (*this) = copy;
}

void Board::make_move(Move move) {
  pStack[pCounter + 1] = pStack[pCounter];
  this->pCounter++;
  pStack[pCounter].make_move(move);
}

void Board::undo_move() { this->pCounter--; }

Board &Board::operator=(Position pos) {
  this->pCounter = 0;
  get_position().BP = pos.BP;
  get_position().WP = pos.WP;
  get_position().K = pos.K;
  get_position().color = pos.color;

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
  const uint64_t color_hash = (p.color == BLACK) ? BLACK_RANDOM : 0;
  uint64_t first =
      static_cast<uint64_t>(p.BP) | (static_cast<uint64_t>(p.WP) << 32);
  uint64_t second = static_cast<uint64_t>(p.K) | (color_hash << 32);
  return hash_combine(hash(first), hash(second));
}

bool Board::is_repetition(int last_rev) const {
  const auto end = std::max(last_rev - 1, 0);
  const auto current = pStack[pCounter];
  for (int i = pCounter - 2; i >= end; i -= 2) {
    if (pStack[i] == current) {
      return true;
    }
  }
  // if we reached the end without encountering an inreversible position
  // we keep looking through the repetition history for 'our ' side

  if (end == 0 && color_us == current.color) {
    for (int i = rep_size - 1; i >= 0; i--) {
      if (rep_history[i] == current) {
        return true;
      }
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
