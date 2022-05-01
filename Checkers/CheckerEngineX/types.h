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

//overload trick

template<typename... Ts>
struct overload : Ts ... {
    using Ts::operator()...;
};
template<typename... Ts> overload(Ts...) -> overload<Ts...>;


template<size_t base> auto power_lambda = [](size_t exp) {
    size_t result = 1;
    for (auto i = 0; i < exp; ++i) {
        result *= base;
    }
    return result;
};


template<size_t size, typename Generator>
constexpr auto get_lut(Generator &&generator) {
    using data_type = decltype(generator(0));
    std::array<data_type, size> result{};

    for (auto i = 0; i < size; ++i) {
        result[i] = generator(i);
    }
    return result;
}


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

inline constexpr auto powers5 = get_lut<8>(power_lambda<5>);

inline constexpr auto powers3 = get_lut<12>(power_lambda<3>);


constexpr int stage_size = 24;

//constant for extensions and reductions


constexpr int prob_cut = 300;
constexpr int sing_ext = 300;
constexpr int asp_wind = 100;
constexpr int MAX_ASP = 2000;


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

enum MoveType {
    PawnMove, KingMove, PromoMove, KingCapture, PawnCapture, PromoCapture
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

template<Color color, PieceType type>
constexpr
uint32_t get_neighbour_squares(uint32_t maske) {

    if constexpr(type == KING) {
        uint32_t squares = defaultShift<color>(maske) | forwardMask<color>(maske);
        squares |= forwardMask<~color>(maske) | defaultShift<~color>(maske);
        return squares;
    } else {
        return defaultShift<color>(maske) | forwardMask<color>(maske);
    }


}

template<Color color>
constexpr
uint32_t get_promotion_rank() {
    if constexpr(color == WHITE) {
        return 0xf0;
    } else {
        return 0x0f000000;
    }
}


#endif //CHECKERSTEST_TYPES_H
