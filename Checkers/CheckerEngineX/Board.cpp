//
//
// Created by Robin on 09.05.2017.
//

#include "Board.h"
#include "types.h"
#include <cstdint>
#include <optional>
std::ofstream debug("debug.txt");
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

  const bool is_not_rev = move.is_capture() || move.is_pawn_move(copy.K);
  if (is_not_rev && copy.color == color_us) {
    debug << "Non Reversible move" << std::endl;
    rep_size = 0;
  } else if (copy.color == color_us) {
    debug << "New entry in repetition history" << std::endl;
    rep_history[rep_size++] = copy;
  }

  debug << "Color_Us : "
        << ((color_us == BLACK)   ? "BLACK"
            : (color_us == WHITE) ? "WHITE"
                                  : " NONE")
        << std::endl;

  debug << "CopyColor : "
        << ((copy.color == BLACK)   ? "BLACK"
            : (copy.color == WHITE) ? "WHITE"
                                    : " NONE")
        << std::endl;

  copy.make_move(move);
  pCounter = 0;
  pStack[pCounter] = copy;
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
  x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 32;
  x *= 0x94d049bb133111ebull;
  x ^= x >> 32;
  x *= 0xff51afd7ed558ccdull;
  x ^= x >> 32;
  return x;
}
/*
uint64_t hash(uint64_t x) {
  x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
  x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
  x = x ^ (x >> 31);
  return x;
}
*/
uint64_t Board::get_current_key() const {
  const Position p = pStack[pCounter];
  const uint64_t color_hash = (p.color == BLACK) ? BLACK_RANDOM : 0;
  uint64_t first =
      static_cast<uint64_t>(p.BP) | (static_cast<uint64_t>(p.WP) << 32);
  uint64_t second = static_cast<uint64_t>(p.K);
  return hash_combine(hash(first), hash(second)) ^ color_hash;
}

bool Board::is_repetition(int last_rev) const {
  const auto end = std::max(last_rev - 1, 0);
  const auto current = pStack[pCounter];
  for (int i = pCounter - 2; i >= end; i -= 2) {
    if (pStack[i] == current) {
      return true;
    }
  }

  if (end == 0 && color_us == current.color) {
    /* debug << "Color_Us : "
           << ((color_us == BLACK)   ? "BLACK"
               : (color_us == WHITE) ? "WHITE"
                                     : " NONE")
           << std::endl;
           */

    for (int i = rep_size - 1; i >= 0; i--) {
      /* debug << "RepHistory : "
             << ((rep_history[i].color == BLACK)   ? "BLACK"
                 : (rep_history[i].color == WHITE) ? "WHITE"
                                                   : " NONE")
             << std::endl;
             */
      if (rep_history[i] == current) {
        debug << "Found a repetition" << std::endl;
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
