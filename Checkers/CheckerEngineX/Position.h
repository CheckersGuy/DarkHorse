/**///
// Created by Robin on 14.01.2018.
//
#include <cstdint>
#include "Move.h"
#include <cassert>
#include "types.h"

#ifndef CHECKERENGINEX_POSITION_H
#define CHECKERENGINEX_POSITION_H

inline uint32_t getHorizontalFlip(uint32_t b) {
    uint32_t x = ((b & MASK_COL_4)) >> 3u;
    x |= (b & MASK_COL_3) >> 1u;
    x |= (b & MASK_COL_1) << 3u;
    x |= (b & MASK_COL_2) << 1u;
    return x;
}

inline uint32_t getVerticalFlip(uint32_t b) {
    uint32_t x = b >> 28u;
    x |= (b >> 20u) & 0xf0u;
    x |= (b >> 12u) & 0xf00u;
    x |= (b >> 4u) & 0xf000u;

    x |= b << 28u;
    x |= (b << 20u) & 0x0f000000u;
    x |= (b << 12u) & 0x00f00000u;
    x |= (b << 4u) & 0x000f0000u;
    return x;
}


inline uint32_t getMirrored(uint32_t b) {
    return getHorizontalFlip(getVerticalFlip(b));
}

struct Position {
    Color color{BLACK};
    uint32_t WP{0}, BP{0}, K{0};
    uint64_t key{0};


    template<Color color>
    uint32_t getCurrent() const {
        if constexpr (color == BLACK)
            return BP;
        else
            return WP;
    }

    template<PieceType type>
    uint32_t getPieces() {
        if constexpr(type == KING) {
            return K;
        } else {
            return (BP | WP) & (~K);
        }
    }

    template<Color color>
    uint32_t getMovers() const {
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
    uint32_t getJumpers() const {
        const uint32_t nocc = ~(BP | WP);
        const uint32_t current = getCurrent<color>();
        const uint32_t opp = getCurrent<~color>();
        const uint32_t kings = current & K;

        uint32_t movers = 0u;
        uint32_t temp = defaultShift<~color>(nocc) & opp;
        if (temp != 0u) {
            movers |= forwardMask<~color>(temp) & current;
        }
        temp = forwardMask<~color>(nocc) & opp;
        if (temp != 0u) {
            movers |= defaultShift<~color>(temp) & current;
        }
        if (kings != 0u) {
            temp = defaultShift<color>(nocc) & opp;
            if (temp != 0u) {
                movers |= forwardMask<color>(temp) & kings;
            }
            temp = forwardMask<color>(nocc) & opp;

            if (temp != 0u) {
                movers |= defaultShift<color>(temp) & kings;
            }
        }
        return movers;
    }

    Color getColor() const;

    bool hasJumps(Color color) const;

    bool isWipe() const;

    bool hasThreat() const;

    bool isEmpty() const;

    std::string position_str() const;

    void makeMove(Move &move);

    void printPosition() const;

    Position getColorFlip() const;

    static Position getStartPosition();


    inline bool operator==(const Position &pos) const {
        return (pos.BP == BP && pos.WP == WP && pos.K == K && pos.color == color);
    }

    inline bool operator!=(const Position &other) const {
        return !(*this == other);
    }


};

std::istream &operator>>(std::istream &stream, const Position &pos);

std::ostream &operator<<(std::ostream &stream, const Position &pos);

#endif //CHECKERENGINEX_POSITION_H
