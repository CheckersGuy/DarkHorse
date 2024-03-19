//
// Created by Robin on 18.12.2017.
//

#ifndef CHECKERSTEST_TYPES_H
#define CHECKERSTEST_TYPES_H

#include <algorithm>
#include <array>
#include <assert.h>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <thread>
// Utility functions and other stuff

enum Result : uint8_t { BLACK_WON = 1, WHITE_WON = 2, DRAW = 3, UNKNOWN = 0 };

inline uint64_t getSystemTime() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

// overload trick

template <typename... Ts> struct overload : Ts... {
  using Ts::operator()...;
};
template <typename... Ts> overload(Ts...) -> overload<Ts...>;

template <size_t base>
auto power_lambda = [](size_t exp) {
  size_t result = 1;
  for (auto i = 0; i < exp; ++i) {
    result *= base;
  }
  return result;
};

template <size_t size, typename Generator>
constexpr auto get_lut(Generator &&generator) {
  using data_type = decltype(generator(0));
  std::array<data_type, size> result{};

  for (auto i = 0; i < size; ++i) {
    result[i] = generator(i);
  }
  return result;
}

constexpr std::array<int, 32> BIT_TO_BOARD = {
    3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,  15, 14, 13, 12,
    19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28};
constexpr std::array<int, 32> BOARD_TO_BIT = {
    3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,  15, 14, 13, 12,
    19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28};

constexpr uint32_t INNER_SQUARES = 132638688;
constexpr uint32_t OUTER_SQUARES = 135796752;
constexpr uint32_t MASK_L3 = 14737632u;
constexpr uint32_t MASK_L5 = 117901063u;
constexpr uint32_t MASK_R3 = 117901056u;
constexpr uint32_t MASK_R5 = 3772834016u;
constexpr uint32_t MASK_COL_1 = 286331153u;
constexpr uint32_t MASK_COL_2 = 572662306u;
constexpr uint32_t MASK_COL_3 = 1145324612u;
constexpr uint32_t MASK_COL_4 = 2290649224u;
constexpr uint32_t PROMO_SQUARES_WHITE = 0xfu;
constexpr uint32_t PROMO_SQUARES_BLACK = 0xf0000000u;
constexpr uint32_t OUTER_RING = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) |
                                (1 << 4) | (1 << 12) | (1 << 20) | (1 << 28) |
                                (1 << 29) | (1 << 30) | (1 << 31) | (1 << 27) |
                                (1 << 19) | (1 << 11);
constexpr uint32_t CENTER = (1 << 9) | (1 << 10) | (1 << 13) | (1 << 14) |
                            (1 << 17) | (1 << 18) | (1 << 22) | (1 << 21);
constexpr uint32_t SINGLE_CORNER = (1 << 28) | (1 << 3);
constexpr uint32_t DOUBLE_CORNER = (1 << 0) | (1 << 4) | (1 << 31) | (1 << 27);
constexpr std::array<uint32_t, 8> MASK_ROWS = {
    0xf, 0xf << 4, 0xf << 8, 0xf << 12, 0xf << 16, 0xf << 20, 0xf << 24};
constexpr std::array<int, 32> PV_LMR_TABLE = {1, 1, 1, 2, 2, 2, 2, 2, 2,
                                              2, 2, 2, 2, 2, 2, 2, 2, 2,
                                              2, 2, 2, 2, 2, 2, 2, 2, 2};

constexpr std::array<int, 32> LMR_TABLE = {1, 1, 1, 2, 2, 3, 3, 3, 3,
                                           3, 3, 3, 3, 3, 3, 3, 3, 3,
                                           3, 3, 3, 3, 3, 3, 3, 3, 3};
constexpr int prob_cut = 24; // 27;
constexpr int asp_wind = 13; // 15;

constexpr int NUM_BUCKETS = 12;
constexpr int MAX_ASP = 200;
constexpr int CORRECTION_SIZE = 512;

constexpr uint64_t BLACK_RANDOM = 7985716234ull;
constexpr uint64_t singular_key = 311234512ull;

using Depth = int;
using Ply = int;
using Value = int;

enum SEARCH : int { MAX_PLY = 256 };
enum Score : int {
  INFINITE = 150000,
  EVAL_INFINITE = 15001,
  MATE_IN_MAX_PLY = 15000 - MAX_PLY,
  MATED_IN_MAX_PLY = -MATE_IN_MAX_PLY,
  TB_WIN = 10000 - MAX_PLY,
  TB_LOSS = -10000 + MAX_PLY,
};

enum Color : char { BLACK = -1, WHITE = 1 };
enum PieceType : int {
  BPAWN = 0,
  WPAWN = 1,
  BKING = 2,
  WKING = 3,
  KING = 4,
  PAWN = 5,
  EMPTY = 6
};
enum Flag : uint8_t { None = 0u, TT_EXACT = 1u, TT_LOWER = 2u, TT_UPPER = 3u };

enum MoveType {
  PawnMove,
  KingMove,
  PromoMove,
  KingCapture,
  PawnCapture,
  PromoCapture
};
enum class TB_RESULT { WIN = 0, LOSS = 1, DRAW = 2, UNKNOWN = 3 };

inline void *std_aligned_alloc(size_t alignment, size_t size) {
#if defined(POSIXALIGNEDALLOC)
  void *mem;
  return posix_memalign(&mem, alignment, size) ? nullptr : mem;
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
  return _mm_malloc(size, alignment);
#elif defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  return std::aligned_alloc(alignment, size);
#endif
}

inline void std_aligned_free(void *ptr) {

#if defined(POSIXALIGNEDALLOC)
  free(ptr);
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
  _mm_free(ptr);
#elif defined(_WIN32)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

inline bool isEval(Value val) { return std::abs(val) < EVAL_INFINITE; }

inline bool isMateVal(Value val) { return std::abs(val) >= MATE_IN_MAX_PLY; }

inline Value loss(int ply) { return -EVAL_INFINITE + ply; }

constexpr Color operator~(Color color) { return static_cast<Color>(-color); }

inline bool isLoss(Value val) { return val <= MATED_IN_MAX_PLY; }

inline bool isWin(Value val) { return val >= MATE_IN_MAX_PLY; }

template <Color color> constexpr uint32_t defaultShift(const uint32_t maske) {
  if constexpr (color == BLACK) {
    return maske << 4u;
  } else {
    return maske >> 4u;
  }
}

template <Color color> constexpr uint32_t forwardMask(const uint32_t maske) {
  if constexpr (color == BLACK) {
    return ((maske & MASK_L3) << 3u) | ((maske & MASK_L5) << 5u);
  } else {
    return ((maske & MASK_R3) >> 3u) | ((maske & MASK_R5) >> 5u);
  }
}

template <Color color, PieceType type>
constexpr uint32_t get_neighbour_squares(uint32_t maske) {
  if constexpr (type == KING) {
    uint32_t squares = defaultShift<color>(maske) | forwardMask<color>(maske);
    squares |= forwardMask<~color>(maske) | defaultShift<~color>(maske);
    return squares;
  } else {
    return defaultShift<color>(maske) | forwardMask<color>(maske);
  }
}

#endif // CHECKERSTEST_TYPES_H
