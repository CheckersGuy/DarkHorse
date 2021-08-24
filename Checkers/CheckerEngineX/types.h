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

constexpr uint32_t big_region = 30583;
constexpr uint32_t region = 13107;
/*
constexpr uint32_t sub_region1 = 12593;
constexpr uint32_t sub_region2 = 17990;
*/
constexpr uint32_t sub_region1 = region;
constexpr uint32_t sub_region2 = region<<1;


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
constexpr std::array<size_t, 8> powers = {1ull, 5ull, 25ull, 125ull, 625ull, 3125ull, 15625ull, 78125ull};
constexpr std::array<size_t, 12> powers3 = {1ull, 3ull, 9ull, 27ull, 81ull, 729ull, 2187ull, 6561ull, 19683ull,
                                            59049ull, 177147ull};

constexpr int stage_size = 24;

//constant for extensions and reductions



constexpr int prob_cut = 300;
constexpr int sing_ext = 300;
constexpr int asp_wind = 100;
constexpr int MAX_ASP = 3000;








/*


constexpr int prob_cut = 30;
constexpr int sing_ext = 50;
constexpr int asp_wind =15;
constexpr int MAX_ASP = 300;
*/






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
