/**/ //
// Created by Robin on 14.01.2018.
//

#ifndef CHECKERENGINEX_POSITION_H
#define CHECKERENGINEX_POSITION_H

#include "Move.h"
#include "types.h"
#include <optional>
#include <random>
#include <sstream>
const uint32_t temp_mask = 0xf;

inline constexpr uint32_t getHorizontalFlip(uint32_t b) {
  uint32_t x = ((b & MASK_COL_4)) >> 3u;
  x |= (b & MASK_COL_3) >> 1u;
  x |= (b & MASK_COL_1) << 3u;
  x |= (b & MASK_COL_2) << 1u;
  return x;
}

inline constexpr uint32_t getVerticalFlip(uint32_t b) {
  uint32_t x = b >> 28u;
  x |= (b >> 20u) & 0xf0u;
  x |= (b >> 12u) & 0xf00u;
  x |= (b >> 4u) & 0xf000u;

  x |= b << 28u;
  x |= (b << 20u) & 0x0f000000u;
  x |= (b << 12u) & 0x00f00000u;
  x |= (b << 4u) & 0x000f0000u;
  return x;
}

inline constexpr uint32_t getMirrored(uint32_t b) {
  return getHorizontalFlip(getVerticalFlip(b));
}

struct Square {
  PieceType type;
  uint32_t index;

  friend std::ostream &operator<<(std::ostream &stream, Square square);
};

struct Position {
  Color color{BLACK};
  uint32_t WP{0u}, BP{0u}, K{0u};

  template <Color color> constexpr uint32_t get_current() const {
    if constexpr (color == BLACK)
      return BP;
    else
      return WP;
  }

  template <Color color, PieceType type> inline uint32_t get_pieces() const {
    if constexpr (color == BLACK && type == KING) {
      return BP & K;
    }
    if constexpr (color == WHITE && type == KING) {
      return WP & K;
    }

    if constexpr (color == BLACK && type == PAWN) {
      return BP & (~K);
    }
    if constexpr (color == WHITE && type == PAWN) {
      return WP & (~K);
    }
  }

  template <Color color> uint32_t get_movers() const {
    const uint32_t nocc = ~(BP | WP);
    const uint32_t current = get_current<color>();
    const uint32_t kings = current & K;

    uint32_t movers =
        (defaultShift<~color>(nocc) | forwardMask<~color>(nocc)) & current;
    if (kings) {
      movers |= (defaultShift<color>(nocc) | forwardMask<color>(nocc)) & kings;
    }
    return movers;
  }

  template <Color color> uint32_t get_attack_squares(uint32_t maske) const {
    const uint32_t nocc = ~(BP | WP);
    const uint32_t current = get_current<color>() & maske;
    const uint32_t kings = current & K;

    uint32_t movers =
        (defaultShift<color>(current) | forwardMask<color>(current));
    movers |= (defaultShift<~color>(kings) | forwardMask<~color>(kings));
    movers &= nocc;
    return movers;
  }

  template <Color color> uint32_t get_jumpers() const {
    const uint32_t nocc = ~(BP | WP);
    const uint32_t current = get_current<color>();
    const uint32_t opp = get_current<~color>();
    const uint32_t kings = current & K;

    uint32_t movers = 0u;
    uint32_t temp = defaultShift<~color>(nocc) & opp;
    if (temp != 0u) {
      movers |= forwardMask<~color>(temp) & current;
    }
    temp = forwardMask<~color>(nocc) & opp;
    if (temp != 0u) {
      movers |= defaultShift<~color>(temp) & current;
    }
    if (kings != 0u) {
      temp = defaultShift<color>(nocc) & opp;
      if (temp != 0u) {
        movers |= forwardMask<color>(temp) & kings;
      }
      temp = forwardMask<color>(nocc) & opp;

      if (temp != 0u) {
        movers |= defaultShift<color>(temp) & kings;
      }
    }
    return movers;
  }

  std::string get_fen_string() const;

  Color get_color() const;

  int piece_count() const;

  template <Color color> bool has_jumps() const {
    return get_jumpers<color>() != 0;
  }

  bool has_jumps(Color color) const;

  bool has_jumps() const;

  bool is_wipe() const;

  bool has_threat() const;

  bool is_empty() const;

  bool is_end() const;

  bool is_legal() const;

  bool has_any_move() const;

  void make_move(Move move);

  void make_move(uint32_t from_index, uint32_t to_index);

  void print_position() const;

  std::string get_pos_string() const;

  Position get_color_flip() const;

  static Position get_start_position();

  static Position pos_from_fen(std::string fen_string);

  inline bool operator==(const Position &pos) const {
    return (pos.BP == BP && pos.WP == WP && pos.K == K && pos.color == color);
  }

  inline bool operator!=(const Position &other) const {
    return !(*this == other);
  }

  friend std::ostream &operator<<(std::ostream &stream, const Position &pos);

  friend std::istream &operator>>(std::istream &stream, Position &pos);

  // given two consecutive positions,returns the move made
  static std::optional<Move> get_move(Position orig, Position next);

  int bucket_index();
};

namespace std {
template <> struct hash<Position> {
  std::array<std::array<uint64_t, 4>, 32> keys;
  uint64_t color_black;
  // should not use the existing zobrist keys

  hash() {
    std::mt19937 generator(23123123ull);
    std::uniform_int_distribution<uint64_t> distrib;
    for (int i = 0; i < 32; ++i) {
      for (int j = 0; j < 4; ++j) {
        const auto value = distrib(generator);
        keys[i][j] = value;
      }
    }
    color_black = distrib(generator);
  }

  uint64_t operator()(const Position &s) const {
    const uint32_t BK = s.K & s.BP;
    const uint32_t WK = s.K & s.WP;
    uint64_t nextKey = 0u;
    uint32_t allPieces = s.BP | s.WP;
    while (allPieces) {
      uint32_t index = Bits::bitscan_foward(allPieces);
      uint32_t maske = 1u << index;
      if ((maske & BK)) {
        nextKey ^= keys[index][BKING];
      } else if ((maske & s.BP)) {
        nextKey ^= keys[index][BPAWN];
      }
      if ((maske & WK)) {
        nextKey ^= keys[index][WKING];
      } else if ((maske & s.WP)) {
        nextKey ^= keys[index][WPAWN];
      }
      allPieces &= allPieces - 1u;
    }
    if (s.color == BLACK) {
      nextKey ^= color_black;
    }
    return nextKey;
  }
};
} // namespace std

#endif // CHECKERENGINEX_POSITION_H
