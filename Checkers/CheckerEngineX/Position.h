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


class Position {
public:
    Color color;
    uint32_t WP, BP, K;
    uint64_t key;

    Position(uint32_t wp, uint32_t bp, uint32_t k, uint64_t cKey) : WP(wp), BP(bp), K(k), key(cKey), color(BLACK) {}

    Position() : WP(0), BP(0), K(0), key(0), color(BLACK) {};

    Position(uint32_t wp, uint32_t bp, uint32_t k) : Position(wp, bp, k, 0) {}

    uint32_t getJumpersBlack();

    uint32_t getJumpersWhite();

    uint32_t getMoversWhite();

    uint32_t getMoversBlack();

    template<Color color> uint32_t getCurrent(){
        if constexpr (color ==BLACK)
            return BP;
        else
            return WP;
    }

    template <Color color> uint32_t getMovers(){
        const uint32_t nocc = ~(BP | WP);
        const uint32_t current =getCurrent<color>();
        const uint32_t kings =current&K;

        uint32_t movers = (defaultShift<~color>(nocc)|forwardMask<~color>(nocc))&current;
        if (kings) {
            movers|=(defaultShift<color>(nocc)|forwardMask<color>(nocc))&kings;
        }
        return movers;
    }

    template<Color color>uint32_t getJumpers() {
        const uint32_t nocc = ~(BP | WP);
        const uint32_t current =getCurrent<color>();
        const uint32_t opp=getCurrent<~color>();
        const uint32_t kings =current&K;

        uint32_t movers = 0;
        uint32_t temp = defaultShift<~color>(nocc) & opp;
        if (temp != 0) {
            movers |= forwardMask<~color>(temp)& current;
        }
        temp = forwardMask<~color>(nocc) & opp;
        if (temp != 0) {
            movers |= defaultShift<~color>(temp)&current;
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

    inline bool operator==(Position &pos) {
        return (pos.BP == BP && pos.WP == WP && pos.K == K && pos.color == color && pos.key == key);
    }
};


#endif //CHECKERENGINEX_POSITION_H
