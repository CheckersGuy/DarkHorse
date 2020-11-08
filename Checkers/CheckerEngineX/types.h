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

//Utility functions and other stuff

inline uint64_t getSystemTime() {
    return std::chrono::duration_cast<std::chrono::milliseconds>
            (std::chrono::high_resolution_clock::now().time_since_epoch()).count();
}

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

constexpr uint32_t LEFT_HALF = 3435973836u;
constexpr uint32_t RIGHT_HALF = LEFT_HALF >> 2u;


constexpr std::array<uint32_t, 32> S = {3u, 2u, 1u, 0u, 7u, 6u, 5u, 4u, 11u, 10u, 9u, 8u, 15u, 14u, 13u, 12u, 19u, 18u,
                                        17u, 16u,
                                        23u, 22u, 21u, 20u, 27u, 26u,
                                        25u, 24u, 31u, 30u, 29u, 28u};

using Depth = int;
using Ply = int;
using Value = int;

enum NodeType {
    PVNode, NONPV
};

enum Score : int {
    INFINITE = 15000000,
    EVAL_INFINITE = INFINITE - 100000,
    NONE = -INFINITE - 1000,
};
enum SEARCH : int {
    MAX_PLY = 256
};
enum Color : int {
    BLACK = -1, WHITE = 1
};
enum PieceType {
    BPAWN = 0, WPAWN = 1, BKING = 2, WKING = 3, KING = 4, PAWN = 5,
};
enum Flag : uint8_t {
    None = 0u, TT_EXACT = 1u, TT_LOWER = 2u, TT_UPPER = 3u
};

inline bool isEval(Value val) {
    return std::abs(val) <= EVAL_INFINITE;
}

inline bool isMateVal(Value val) {
    return std::abs(val) >= INFINITE - MAX_PLY && std::abs(val) <= INFINITE;
}

inline int getMateInX(Value val) {
    if (isMateVal(val)) {
        return INFINITE - std::abs(val);
    } else {
        return INFINITE;
    }
}

inline Value loss(int ply) {
    return -INFINITE + ply;
}

constexpr Color operator~(Color color) {
    return static_cast<Color>(-color);
}

inline bool isLoss(Value val) {
    return val <= -INFINITE + MAX_PLY;
}

inline bool isWin(Value val) {
    return val >= INFINITE - MAX_PLY;
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


inline Value clampScore(Value val) {
    if (isLoss(val)) {
        return -INFINITE;
    } else if (isWin(val)) {
        return INFINITE;
    }
    return val;
}

inline Value addSafe(Value val, Value incre) {
    return clampScore(val + incre);
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
