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
            (std::chrono::steady_clock::now().time_since_epoch()).count();
}



constexpr uint32_t LEFT_HALF = 3435973836ull;
constexpr uint32_t RIGHT_HALF = LEFT_HALF >> 2;
constexpr uint32_t MASK_L3 = 14737632ull;
constexpr uint32_t MASK_L5 = 117901063ull;
constexpr uint32_t MASK_R3 = 117901056ull;
constexpr uint32_t MASK_R5 = 3772834016ull;
constexpr uint32_t MASK_COL_1=286331153ull;
constexpr uint32_t MASK_COL_2=572662306ull;
constexpr uint32_t MASK_COL_3=1145324612ull;
constexpr uint32_t MASK_COL_4=2290649224ull;




using Depth =int;



enum  NodeType {
    PVNode, NONPV
};

enum Score {
    WHITE_WIN = 15500,
    WHITE_LOSS = -15500,
    BLACK_WIN = -15500,
    BLACK_LOSS = 15500,
    DRAW = 0,
    INFINITE = 1500000,
    EASY_MOVE = 99999999,
    INVALID = 100000000

};
enum  SEARCH {
    MAX_PLY = 128, MAX_MOVE = 320, DRAW_RULE = 50, ONE_PLY = 1000
};
enum Color {
    BLACK = -1, WHITE = 1, NONE
};
enum PieceType {
    BPAWN = 0, WPAWN = 1, BKING = 2, WKING = 3, KING = 4, PAWN = 5,
};
enum Flag : uint16_t {
    TT_EXACT = 1, TT_LOWER = 2, TT_UPPER = 3, UNKNOWN = 555
};


class Value {
public:
    int value = 0;

    static Value loss(Color color, int ply);

    constexpr Value(int value):value(value){};

    constexpr Value();

    bool isWinning() const;

    bool isBlackWin() const;

    bool isWhiteWin() const;

    bool isEasyMove() const;

    Value valueFromTT(int ply);

    Value toTT(int ply);

    bool isEval() const;

    bool isInRange(int a, int b) const;

    template<class T>bool isInRange(T a, T b) const;

    constexpr Value operator=(int value);

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

    template<typename T> T as(){
        return static_cast<T>(value);
    }
};

inline Value &Value::operator++() {
    this->value++;
    return *this;
}

inline Value Value::operator++(int) {
    Value copy(this->value);
    this->value++;
    return copy;
}

inline Value &Value::operator--() {
    this->value--;
    return *this;
}

inline Value Value::operator--(int) {
    Value copy(this->value);
    this->value--;
    return copy;
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
    } else if (color == BLACK) {
        return -(WHITE_WIN - ply);
    }
}

constexpr Value::Value() {}

//operators
constexpr Value operator~(Value val) {
    Value next = -val.value;
    return next;
}

constexpr Color operator~(Color color) {
    return (color == BLACK) ? WHITE : BLACK;
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


constexpr Value Value::operator=(int value) {
    this->value = value;
    return *this;
}

constexpr Value operator+(const Value val, const Value val2) {
    Value next;
    next.value=val.value + val2.value;
    return next;
}

constexpr Value operator+(Value val, int val2) {
    Value next ;
    next.value=val.value+val2;
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

inline bool Value::isWinning() const {
    assert(value >= -INFINITE && value <= INFINITE);
    if (std::abs(this->value) + MAX_PLY >= WHITE_WIN) {
        return true;
    }
    return false;
}

inline bool Value::isBlackWin() const {
    assert(value >= -INFINITE && value <= INFINITE);
    if (this->value - MAX_PLY <= BLACK_WIN) {
        return true;
    }
    return false;
}

inline bool Value::isWhiteWin() const {
    assert(value >= -INFINITE && value <= INFINITE);
    if (this->value + MAX_PLY >= WHITE_WIN) {
        return true;
    }
    return false;
}

//inclusive
template<class T>inline bool Value::isInRange(T a,  T b) const {
    return (this->value >= a && this->value <= b);
}


inline Value Value::valueFromTT(int ply) {
    Value result = this->value;
    if (abs(this->value) + MAX_PLY >= WHITE_WIN) {
        if (this->value < 0) {
            result += ply;
        } else {
            result -= ply;
        }
    }
    return result;
}

inline Value Value::toTT(int ply) {
    Value result = this->value;
    if (result - MAX_PLY <= BLACK_WIN) {
        result -= ply;
    } else if (result + MAX_PLY >= WHITE_WIN) {
        result += ply;
    }
    return result;
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
        return maske << 4;
    } else {
        return maske >> 4;
    }
}

template<Color color>
inline
uint32_t forwardMask(const uint32_t maske) {
    if constexpr (color == BLACK) {
        return ((maske & MASK_L3) << 3) | ((maske & MASK_L5) << 5);
    } else {
        return ((maske & MASK_R3) >> 3) | ((maske & MASK_R5) >> 5);
    }
}


#endif //CHECKERSTEST_TYPES_H
