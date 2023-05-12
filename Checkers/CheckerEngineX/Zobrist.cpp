//
// Created by Robin on 06.05.2018.
//
#include "Zobrist.h"

namespace Zobrist {
std::array<std::array<uint64_t, 4>, 32> ZOBRIST_KEYS;
uint64_t color_black = 0ull;
uint64_t skip_hash = 0ull;
uint64_t seed = 0x312341241ull;
std::mt19937_64 generator;

void init_zobrist_keys(uint64_t s) {
  generator = std::mt19937_64(s);
  seed = s;
  std::uniform_int_distribution<uint64_t> distrib;
  for (int i = 0; i < 32; ++i) {
    for (int j = 0; j < 4; ++j) {
      const auto value = distrib(generator);
      ZOBRIST_KEYS[i][j] = value;
    }
  }
  color_black = distrib(generator);
  skip_hash = distrib(generator);
}

void init_zobrist_keys() {
  constexpr uint64_t seed = 1070372ull;
  init_zobrist_keys(seed);
}

uint64_t generate_key(const Position &pos) {
  const uint32_t BK = pos.K & pos.BP;
  const uint32_t WK = pos.K & pos.WP;
  uint64_t nextKey = 0u;
  uint32_t allPieces = pos.BP | pos.WP;
  while (allPieces) {
    uint32_t index = Bits::bitscan_foward(allPieces);
    uint32_t maske = 1u << index;
    if ((maske & BK)) {
      nextKey ^= ZOBRIST_KEYS[index][BKING];
    } else if ((maske & pos.BP)) {
      nextKey ^= ZOBRIST_KEYS[index][BPAWN];
    }
    if ((maske & WK)) {
      nextKey ^= ZOBRIST_KEYS[index][WKING];
    } else if ((maske & pos.WP)) {
      nextKey ^= ZOBRIST_KEYS[index][WPAWN];
    }
    allPieces &= allPieces - 1u;
  }
  if (pos.color == BLACK) {
    nextKey ^= color_black;
  }
  return nextKey;
}

void update_zobrist_keys(Position &pos, Move move) {
  const int pawn_index = (pos.get_color() == BLACK) ? BPAWN : WPAWN;
  const int king_index = (pos.get_color() == BLACK) ? BKING : WKING;
  const int opp_king_index = (pos.get_color() == BLACK) ? WKING : BKING;
  const int opp_pawn_index = (pos.get_color() == BLACK) ? WPAWN : BPAWN;
  auto toIndex = move.get_to_index();
  auto fromIndex = move.get_from_index();
  constexpr uint32_t promo_squares = PROMO_SQUARES_WHITE | PROMO_SQUARES_BLACK;
  if (((move.from & pos.K) == 0u) && (move.to & promo_squares) != 0u) {
    pos.key ^= Zobrist::ZOBRIST_KEYS[toIndex][king_index];
    pos.key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pawn_index];
  } else if (((move.from & pos.K) != 0u)) {
    pos.key ^= Zobrist::ZOBRIST_KEYS[toIndex][king_index];
    pos.key ^= Zobrist::ZOBRIST_KEYS[fromIndex][king_index];
  } else {
    pos.key ^= Zobrist::ZOBRIST_KEYS[toIndex][pawn_index];
    pos.key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pawn_index];
  }
  uint32_t captures = move.captures;
  while (captures) {
    const uint32_t index = Bits::bitscan_foward(captures);
    if (((1u << index) & move.captures & pos.K) != 0) {
      pos.key ^= Zobrist::ZOBRIST_KEYS[index][opp_king_index];
    } else {
      pos.key ^= Zobrist::ZOBRIST_KEYS[index][opp_pawn_index];
    }
    captures &= captures - 1u;
  }

  pos.key ^= Zobrist::color_black;
}

uint64_t get_move_key(Position &cur_pos, Move move) {
  uint64_t move_key = 0ull;
  const int pawn_index = (cur_pos.get_color() == BLACK) ? BPAWN : WPAWN;
  const int king_index = (cur_pos.get_color() == BLACK) ? BKING : WKING;
  const int opp_king_index = (cur_pos.get_color() == BLACK) ? WKING : BKING;
  const int opp_pawn_index = (cur_pos.get_color() == BLACK) ? WPAWN : BPAWN;
  auto toIndex = move.get_to_index();
  auto fromIndex = move.get_from_index();
  constexpr uint32_t promo_squares = PROMO_SQUARES_WHITE | PROMO_SQUARES_BLACK;
  if (((move.from & cur_pos.K) == 0u) && (move.to & promo_squares) != 0u) {
    move_key ^= Zobrist::ZOBRIST_KEYS[toIndex][king_index];
    move_key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pawn_index];
  } else if (((move.from & cur_pos.K) != 0u)) {
    move_key ^= Zobrist::ZOBRIST_KEYS[toIndex][king_index];
    move_key ^= Zobrist::ZOBRIST_KEYS[fromIndex][king_index];
  } else {
    move_key ^= Zobrist::ZOBRIST_KEYS[toIndex][pawn_index];
    move_key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pawn_index];
  }
  uint32_t captures = move.captures;
  while (captures) {
    const uint32_t index = Bits::bitscan_foward(captures);
    if (((1u << index) & move.captures & cur_pos.K) != 0) {
      move_key ^= Zobrist::ZOBRIST_KEYS[index][opp_king_index];
    } else {
      move_key ^= Zobrist::ZOBRIST_KEYS[index][opp_pawn_index];
    }
    captures &= captures - 1u;
  }
  return move_key;
}

} // namespace Zobrist
