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

constexpr uint32_t MASK_L3 = 14737632;
constexpr uint32_t MASK_L5 = 117901063;
constexpr uint32_t MASK_R3 = 117901056;
constexpr uint32_t MASK_R5 = 3772834016;


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
