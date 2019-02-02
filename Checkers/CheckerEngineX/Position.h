//
// Created by Robin on 14.01.2018.
//
#include "stdint.h"
#include "Move.h"
#include <cassert>
#include "types.h"
#include "immintrin.h"

#ifndef CHECKERENGINEX_POSITION_H
#define CHECKERENGINEX_POSITION_H


inline uint32_t getHorizontalFlip(uint32_t b) {
    uint32_t x = (b&MASK_COL_4)>>3;
    x|= (b&MASK_COL_3)>>1;
    x|= (b&MASK_COL_1)<<3;
    x|= (b&MASK_COL_2)<<1;
    return x;
}

inline uint32_t getVerticalFlip(uint32_t b) {
    uint32_t x = b >> 28;
    x |= (b >> 20) & 0xf0;
    x |= (b >> 12) & 0xf00;
    x |= (b >> 4) & 0xf000;

    x |= b << 28;
    x |= (b << 20) & 0x0f000000;
    x |= (b << 12) & 0x00f00000;
    x |= (b << 4) & 0x000f0000;
    return x;
}

inline uint32_t getMirrored(uint32_t b) {
    return getHorizontalFlip(getVerticalFlip(b));
}

class Position {
public:
    Color color;
    uint32_t WP, BP, K;
    uint64_t key;

    Position(uint32_t wp, uint32_t bp, uint32_t k, uint64_t cKey) : WP(wp), BP(bp), K(k), key(cKey), color(BLACK) {}

    Position() : WP(0), BP(0), K(0), key(0), color(BLACK) {};

    Position(uint32_t wp, uint32_t bp, uint32_t k) : Position(wp, bp, k, 0) {}

    template<Color color>
    uint32_t getCurrent() {
        if constexpr (color == BLACK)
            return BP;
        else
            return WP;
    }

    template<Color color>
    uint32_t getMovers() {
        const uint32_t nocc = ~(BP | WP);
        const uint32_t current = getCurrent<color>();
        const uint32_t kings = current & K;

        uint32_t movers = (defaultShift<~color>(nocc) | forwardMask<~color>(nocc)) & current;
        if (kings) {
            movers |= (defaultShift<color>(nocc) | forwardMask<color>(nocc)) & kings;
        }
        return movers;
    }

    template<Color color>
    uint32_t getJumpers() {
        const uint32_t nocc = ~(BP | WP);
        const uint32_t current = getCurrent<color>();
        const uint32_t opp = getCurrent<~color>();
        const uint32_t kings = current & K;

        uint32_t movers = 0;
        uint32_t temp = defaultShift<~color>(nocc) & opp;
        if (temp != 0) {
            movers |= forwardMask<~color>(temp) & current;
        }
        temp = forwardMask<~color>(nocc) & opp;
        if (temp != 0) {
            movers |= defaultShift<~color>(temp) & current;
        }
        if (kings != 0) {
            temp = defaultShift<color>(nocc) & opp;
            if (temp != 0) {
                movers |= forwardMask<color>(temp) & kings;
            }
            temp = forwardMask<color>(nocc) & opp;

            if (temp != 0) {
                movers |= defaultShift<color>(temp) & kings;
            }
        }
        return movers;
    }

    Color getColor();

    bool hasJumps(Color color);

    bool isLoss();

    bool isWipe();

    bool hasThreat();

    bool isEmpty();

    void makeMove(Move move);

    void undoMove(Move move);

    void printPosition();

    Position getColorFlip();



    inline bool operator==(const Position &pos) {
        return (pos.BP == BP && pos.WP == WP && pos.K == K && pos.color == color && pos.key == key);
    }

    inline bool operator!=(const Position &other) {
        return !(*this == other);
    }


};

std::istream &operator>>(std::istream &stream, const Position &pos);

std::ostream &operator<<(std::ostream &stream, const Position &pos);

#endif //CHECKERENGINEX_POSITION_H
