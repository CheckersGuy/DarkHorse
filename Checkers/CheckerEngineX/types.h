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

constexpr int prob_cut = 35; // 30;
constexpr int asp_wind = 15; // 15;
constexpr int NUM_BUCKETS = 4;
constexpr int MAX_ASP = 200;
constexpr int MAX_KILLERS = 2;

using Depth = int;
using Ply = int;
using Value = int;

enum SEARCH : int { MAX_PLY = 256 };
enum Score : int {
  INFINITE = 150000,
  EVAL_INFINITE = 15001,
  MATE_IN_MAX_PLY = 15000 - MAX_PLY,
  MATED_IN_MAX_PLY = -MATE_IN_MAX_PLY,
  TB_WIN = 13000 - MAX_PLY,
  TB_LOSS = -TB_WIN
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

inline bool isEval(Value val) { return std::abs(val) <= EVAL_INFINITE; }

inline bool isMateVal(Value val) { return std::abs(val) >= MATE_IN_MAX_PLY; }

inline Value loss(int ply) { return -EVAL_INFINITE + ply; }

constexpr Color operator~(Color color) { return static_cast<Color>(-color); }

inline bool isLoss(Value val) { return val <= MATED_IN_MAX_PLY; }

inline bool isWin(Value val) { return val >= MATE_IN_MAX_PLY; }

inline Value value_to_tt(Value v, int ply) {
  return v >= MATE_IN_MAX_PLY ? v + ply : v <= MATED_IN_MAX_PLY ? v - ply : v;
}

inline Value value_from_tt(Value v, int ply) {
  return v >= MATE_IN_MAX_PLY ? v - ply : v <= MATED_IN_MAX_PLY ? v + ply : v;
}

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
