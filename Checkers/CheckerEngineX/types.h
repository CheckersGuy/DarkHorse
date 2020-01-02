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

constexpr uint32_t S[32] = {3u, 2u, 1u, 0u, 7u, 6u, 5u, 4u, 11u, 10u, 9u, 8u, 15u, 14u, 13u, 12u, 19u, 18u, 17u, 16u,
                            23u, 22u, 21u, 20u, 27u, 26u,
                            25u, 24u, 31u, 30u, 29u, 28u};

using Depth =int;
using Ply =int;

constexpr int scalfac=128;
enum NodeType {
    PVNode, NONPV
};

enum Score : int {
    WHITE_WIN = 1550000,
    WHITE_LOSS = -1550000,
    BLACK_WIN = -1550000,
    BLACK_LOSS = 1550000,
    DRAW = 0,
    INFINITE = 15000000,
    EASY_MOVE = 99999999,
    INVALID = 100000000

};
enum SEARCH {
    MAX_PLY = 128, MAX_MOVE = 320, ONE_PLY = 1000
};
enum Color : int {
    BLACK = -1, WHITE = 1, NONE
};
enum PieceType {
    BPAWN = -1, WPAWN = 1, BKING = -2, WKING = 2, KING = 4, PAWN = 5,
};
enum Flag : uint8_t {
    TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3
};


struct Value {
    int value{0};

    static Value loss(Color color, int ply);

    constexpr Value(int value) noexcept : value(value) {};

    constexpr Value() = default;

    bool isWinning() const;

    bool isBlackWin() const;

    bool isWhiteWin() const;

    bool isEasyMove() const;

    Value valueFromTT(int ply);

    Value toTT(int ply);

    bool isEval() const;

    template<class T, class E>
    bool isInRange(T a, E b) const {
        return this->value >= a && this->value <= b;
    }

    constexpr Value &operator=(int value);

    Value &operator+=(Value other);

    Value &operator+=(int other);

    Value &operator-=(Value other);

    Value &operator-=(int other);

    Value &operator*=(Value other);

    Value &operator*=(int other);

    //pre increment
    Value &operator++();

    //postincrement
    Value operator++(int);

    //pre increment
    Value &operator--();

    //post increment
    Value operator--(int);

    template<typename T>
    T as() {
        return static_cast<T>(value);
    }
};

inline Value &Value::operator++() {
    this->value++;
    return *this;
}

inline Value Value::operator++(int) {
    return this->value + 1;;
}

inline Value &Value::operator--() {
    this->value--;
    return *this;
}

inline Value Value::operator--(int) {
    return this->value - 1;
}

inline Value &Value::operator+=(Value other) {
    this->value += other.value;
    return *this;
}

inline Value &Value::operator+=(int other) {
    this->value += other;
    return *this;
}

inline Value &Value::operator-=(Value other) {
    this->value -= other.value;
    return *this;
}

inline Value &Value::operator-=(int other) {
    this->value -= other;
    return *this;
}

inline Value &Value::operator*=(Value other) {
    this->value *= other.value;
    return *this;
}

inline Value &Value::operator*=(int other) {
    this->value *= other;
    return *this;
}

inline bool Value::isEval() const {
    return isInRange(-INFINITE, INFINITE);
}

inline Value Value::loss(Color color, int ply) {
    if (color == WHITE) {
        return BLACK_WIN + ply;
    } else {
        return -(WHITE_WIN - ply);
    }
}

//operators
constexpr Value operator~(Value val) {
    Value next = -val.value;
    return next;
}

constexpr Color operator~(Color color) {
    if (color == BLACK) {
        return WHITE;
    } else {
        return BLACK;
    }
}

constexpr bool operator!=(Value one, Value two) {
    return one.value != two.value;
}

constexpr bool operator==(Value one, Value two) {
    return one.value == two.value;
}

constexpr bool operator==(Value one, int two) {
    return one.value == two;
}

constexpr bool operator!=(Value one, int two) {
    return one.value != two;
}

constexpr bool operator!=(int one, Value two) {
    return one != two.value;
}


constexpr Value &Value::operator=(int val) {
    this->value = val;
    return *this;
}


constexpr Value operator+(const Value val, const Value val2) {
    Value next;
    next.value = val.value + val2.value;
    return next;
}

constexpr Value operator+(Value val, int val2) {
    Value next;
    next.value = val.value + val2;
    return next;
}

constexpr Value operator*(Value val, Value val2) {
    Value next = Value(val.value * val2.value);
    return next;
}

constexpr Value operator*(Value val, int val2) {
    Value next = Value(val.value * val2);
    return next;
}

constexpr Value operator-(Value val, Value val2) {
    Value next = Value(val.value - val2.value);
    return next;
}

constexpr Value operator-(Value val, int val2) {
    Value next = Value(val.value - val2);
    return next;
}

constexpr bool operator>(Value val1, Value val2) {
    return val1.value > val2.value;
}

constexpr bool operator>=(Value val1, Value val2) {
    return val1.value >= val2.value;
}

constexpr bool operator<(Value val1, Value val2) {
    return val1.value < val2.value;
}

constexpr bool operator<=(Value val1, Value val2) {
    return val1.value <= val2.value;
}


constexpr bool operator>(Value val1, int val2) {
    return val1.value > val2;
}

constexpr bool operator>=(Value val1, int val2) {
    return val1.value >= val2;
}

constexpr bool operator<(Value val1, int val2) {
    return val1.value < val2;
}

constexpr bool operator<=(Value val1, int val2) {
    return val1.value <= val2;
}


//Value class-functions
inline bool Value::isEasyMove() const {
    return (this->value == EASY_MOVE);
}

inline bool Value::isBlackWin() const {
    assert(value >= -INFINITE && value <= INFINITE);
    return this->value - MAX_PLY <= BLACK_WIN;
}

inline bool Value::isWhiteWin() const {
    assert(value >= -INFINITE && value <= INFINITE);
    return this->value + MAX_PLY >= WHITE_WIN;
}

inline bool Value::isWinning() const {
    assert(value >= -INFINITE && value <= INFINITE);
    return isWhiteWin() || isBlackWin();
}

inline Value Value::valueFromTT(int ply) {
    if (this->value + MAX_PLY >= WHITE_WIN) {
        return value + ply;
    }
    if (this->value - MAX_PLY <= BLACK_WIN) {
        return value - ply;
    }
    return *this;
}

inline Value Value::toTT(int ply) {
    if (this->value - MAX_PLY <= BLACK_WIN) {
        return value - ply;
    } else if (this->value + MAX_PLY >= WHITE_WIN) {
        return value + ply;
    }
    return *this;
}

inline std::ostream &operator<<(std::ostream &outstream, const Value val) {
    outstream << val.value;
    return outstream;
}

inline Value clampScore(Value val) {
    //Scores are only positive
    if (val >= (WHITE_WIN + MAX_PLY)) {
        return Value(WHITE_WIN + MAX_PLY);
    } else if (val <= BLACK_WIN - MAX_PLY) {
        return Value(BLACK_WIN - MAX_PLY);
    }
    return val;
}

inline Value addSafe(Value val, Value incre) {
    return clampScore(val + incre);
}

template<Color color>
inline
uint32_t defaultShift(const uint32_t maske) {
    if constexpr(color == BLACK) {
        return maske << 4u;
    } else {
        return maske >> 4u;
    }
}

template<Color color>
inline
uint32_t forwardMask(const uint32_t maske) {
    if constexpr (color == BLACK) {
        return ((maske & MASK_L3) << 3u) | ((maske & MASK_L5) << 5u);
    } else {
        return ((maske & MASK_R3) >> 3u) | ((maske & MASK_R5) >> 5u);
    }
}


#endif //CHECKERSTEST_TYPES_H
