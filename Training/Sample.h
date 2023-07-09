//
// Created by robin on 01.09.21.
//

#ifndef READING_SAMPLE_H
#define READING_SAMPLE_H

#include "Position.h"
#include <fstream>
#include <random>
enum Result : uint8_t { BLACK_WON = 1, WHITE_WON = 2, DRAW = 3, UNKNOWN = 0 };

std::string result_to_string(Result result);

constexpr Result operator~(Result one) {
  if (one == BLACK_WON)
    return WHITE_WON;
  if (one == WHITE_WON)
    return BLACK_WON;
  return one;
}

struct Sample {
  Position position;
  Result result{UNKNOWN};
  int move{-1};
  friend std::ofstream &operator<<(std::ofstream &stream, const Sample s);

  friend std::ifstream &operator>>(std::ifstream &stream, const Sample &s);
  friend std::ostream &operator<<(std::ostream &stream, const Sample &s) {
    s.position.print_position();
    stream << s.position.get_fen_string() << std::endl;
    stream << "Color: " << ((s.position.color == BLACK) ? "BLACK" : "WHITE")
           << std::endl;
    stream << "Move: " << s.move << std::endl;
    std::cout << "Result: " << s.result << std::endl;
    if (s.result == DRAW) {
      stream << "DRAW";
    } else if (s.result == BLACK_WON)
      stream << "BLACK_WON";
    else if (s.result == WHITE_WON)
      stream << "WHITE_WON";
    else if (s.result == UNKNOWN) {
      stream << "UNKNOWN";
    }

    stream << std::endl;

    return stream;
  }

  bool operator==(const Sample &other) const;

  bool operator!=(const Sample &other) const;

  bool is_training_sample() const;
};

namespace std {

template <> struct hash<Sample> {
  std::hash<int> hasher;
  std::array<std::array<uint64_t, 4>, 32> keys;
  uint64_t color_hash, draw_hash, win_hash, loss_hash;
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
    color_hash = distrib(generator);
    loss_hash = distrib(generator);
    draw_hash = distrib(generator);
    win_hash = distrib(generator);
  }

  uint64_t operator()(const Sample &s) const {
    const uint32_t BK = s.position.K & s.position.BP;
    const uint32_t WK = s.position.K & s.position.WP;
    uint64_t nextKey = 0u;
    uint32_t allPieces = s.position.BP | s.position.WP;
    while (allPieces) {
      uint32_t index = Bits::bitscan_foward(allPieces);
      uint32_t maske = 1u << index;
      if ((maske & BK)) {
        nextKey ^= keys[index][BKING];
      } else if ((maske & s.position.BP)) {
        nextKey ^= keys[index][BPAWN];
      }
      if ((maske & WK)) {
        nextKey ^= keys[index][WKING];
      } else if ((maske & s.position.WP)) {
        nextKey ^= keys[index][WPAWN];
      }
      allPieces &= allPieces - 1u;
    }
    if (s.position.color == BLACK) {
      nextKey ^= color_hash;
    }
    if (s.result == DRAW)
      nextKey = nextKey ^ draw_hash;
    if (s.result == WHITE_WON)
      nextKey = nextKey ^ win_hash;
    if (s.result == BLACK_WON)
      nextKey = nextKey ^ loss_hash;

    return nextKey;
  }
};

} // namespace std

#endif // READING_SAMPLE_H
