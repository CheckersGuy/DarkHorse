//
// Created by Robin on 18.12.2017.
//

#ifndef CHECKERSTEST_TYPES_H
#define CHECKERSTEST_TYPES_H


#include <iostream>
#include <cstdint>
#include <algorithm>
#include <cassert>
#include <thread>
#include <array>

//Utility functions and other stuff

inline uint64_t getSystemTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}


constexpr auto powc = [](size_t base, size_t power) {
    size_t res = 1ull;
    for (size_t i = 0; i < power; ++i) {
        res *= base;
    }
    return res;
};


constexpr uint32_t big_region = 30583;
constexpr uint32_t region = 13107;
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
constexpr std::array<size_t, 8> powers5 = {powc(5, 0), powc(5, 1), powc(5, 2), powc(5, 3), powc(5, 4), powc(5, 5),
                                           powc(5, 6), powc(5, 7)};
constexpr std::array<size_t, 12> powers3 = {powc(3, 0), powc(3, 1), powc(3, 2), powc(3, 3), powc(3, 4), powc(3, 5),
                                            powc(3, 6), powc(3, 7), powc(3, 8),
                                            powc(3, 9), powc(3, 10), powc(3, 11)};

constexpr int stage_size = 24;

//constant for extensions and reductions




constexpr int prob_cut = 300;
constexpr int sing_ext = 300;
constexpr int asp_wind = 100;
constexpr int MAX_ASP = 5000;








using Depth = int;
using Ply = int;
using Value = int;


enum Score : int {
    INFINITE = 15000000,
    EVAL_INFINITE = INFINITE - 100000,
};
enum SEARCH : int {
    MAX_PLY = 256
};
enum Color : int {
    BLACK = -1, WHITE = 1
};
enum PieceType : uint8_t {
    BPAWN = 0, WPAWN = 1, BKING = 2, WKING = 3, KING = 4, PAWN = 5, EMPTY = 6
};
enum Flag : uint8_t {
    None = 0u, TT_EXACT = 1u, TT_LOWER = 2u, TT_UPPER = 3u
};

inline bool isEval(Value val) {
    return std::abs(val) <= EVAL_INFINITE;
}

inline bool isMateVal(Value val) {
    return std::abs(val) >= EVAL_INFINITE && std::abs(val) < INFINITE;
}


inline Value loss(int ply) {
    return -INFINITE + ply;
}

constexpr Color operator~(Color color) {
    return static_cast<Color>(-color);
}

inline bool isLoss(Value val) {
    return val <= -EVAL_INFINITE;
}

inline bool isWin(Value val) {
    return val >= EVAL_INFINITE;
}

inline Value valueFromTT(Value val, int ply) {
    if (isLoss(val)) {
        return val + ply;
    } else if (isWin(val)) {
        return val - ply;
    }
    return val;
}

inline Value toTT(Value val, int ply) {
    if (isLoss(val)) {
        return val - ply;
    } else if (isWin(val)) {
        return val + ply;
    }
    return val;
}

inline int div_round(int a, int b) {
    a += b / 2;
    const int div = a / b;
    return (a < 0 && a != b * div) ? div - 1 : div;
}


template<Color color>
constexpr uint32_t defaultShift(const uint32_t maske) {
    if constexpr(color == BLACK) {
        return maske << 4u;
    } else {
        return maske >> 4u;
    }
}

template<Color color>
constexpr
uint32_t forwardMask(const uint32_t maske) {
    if constexpr (color == BLACK) {
        return ((maske & MASK_L3) << 3u) | ((maske & MASK_L5) << 5u);
    } else {
        return ((maske & MASK_R3) >> 3u) | ((maske & MASK_R5) >> 5u);
    }
}


#endif //CHECKERSTEST_TYPES_H
