//
// Created by Robin on 10.06.2017.
//

#ifndef CHECKERSTEST_GAMEEVALUATION_H
#define CHECKERSTEST_GAMEEVALUATION_H

//#include "intrin.h"
#include "Board.h"
#include "iostream"
#include <stdio.h>

constexpr uint32_t LEFT_HALF = 3435973836;
constexpr uint32_t RIGHT_HALF = LEFT_HALF>>2;
const uint32_t MASK_DCORNER_UP = (1 << S[31]);
const uint32_t MASK_DCORNER_DOWN = (1 << S[0]);
constexpr uint32_t CENTER_KING = 6710784;
const uint32_t MASK_ROW0 = (1 << S[0]) | (1 << S[1]) | (1 << S[2]) | (1 << S[3]);
const uint32_t MASK_ROW7 = (1 << S[31]) | (1 << S[30]) | (1 << S[29]) | (1 << S[28]);
const uint32_t DIAGONAL =
        (1 << S[0]) | (1 << S[4]) | (1 << S[9]) | (1 << S[13]) | (1 << S[18]) | (1 << S[22]) | (1 << S[27]) |
        (1 << S[31]);

//#define USE_END_GAME






Value evaluate(Board &board);


inline int balanceScore(Position &pos) {
    uint32_t BP=pos.BP&(~pos.K);
    uint32_t WP=pos.WP&(~pos.K);

    int scoreWhite = std::abs(_popcnt32(WP & LEFT_HALF) - _popcnt32(WP & RIGHT_HALF));
    int scoreBlack = std::abs(_popcnt32(BP & LEFT_HALF) - _popcnt32(BP & RIGHT_HALF));
    return -2*(scoreWhite - scoreBlack);
}


inline int materialEvaluation(Position &pos) {

    int score = 0;
    if (pos.K != 0) {
        const int WK = _popcnt32(pos.K & (pos.WP));
        const int BK = _popcnt32(pos.K & (pos.BP));
        score += 140 * (WK - BK);
    }
    const uint32_t nKings =~pos.K;
    const int WP = _popcnt32(pos.WP&nKings);
    const int BP = _popcnt32(pos.BP&nKings);
    score += 100 * (WP - BP);
    return score;
}

inline int saveSquares(Position &pos) {
    const uint32_t BK =pos.BP&pos.K;
    const uint32_t WK =pos.WP&pos.K;
    const uint32_t BP =pos.BP&(~pos.K);
    const uint32_t WP =pos.WP&(~pos.K);

    const int kingScore =(_popcnt32(WK & MASK_ROW7)- _popcnt32(BK & MASK_ROW0))*2;
    const int pawnScore =(_popcnt32(WP & MASK_ROW7)- _popcnt32(BP & MASK_ROW0))*4;


    return kingScore+pawnScore;
}

inline int holeScore(Position pos) {
    int score = 0;
    const uint32_t BP=pos.BP&(~pos.K);
    const uint32_t WP=pos.WP&(~pos.K);

    if ((WP & (1 << S[7])) != 0 && (BP & (1 << S[3])) != 0) {
        score -= 50;
    }
    if ((BP & (1 << S[24])) != 0 && (WP & (1 << S[28])) != 0) {
        score += 50;
    }

    return score;
}


inline int diagonalSquares(Position &pos) {
    const uint32_t BK =pos.BP&pos.K;
    const uint32_t WK =pos.WP&pos.K;
    const uint32_t BP =pos.BP&(~pos.K);
    const uint32_t WP =pos.WP&(~pos.K);

    const int kingScore =(_popcnt32(WK & DIAGONAL)- _popcnt32(BK & DIAGONAL))*4;
    const int pawnScore =(_popcnt32(WP & DIAGONAL)- _popcnt32(BP & DIAGONAL))*2;


    return (kingScore+pawnScore);
}


inline int cornerScore(Position pos) {
    const uint32_t BK = pos.K & pos.BP;
    const uint32_t WK = pos.K & pos.WP;
    int score = 0;
    if ((BK & MASK_DCORNER_UP) == MASK_DCORNER_UP || (BK & MASK_DCORNER_DOWN) == MASK_DCORNER_DOWN) {
        score += 40;
    }
    if ((WK & MASK_DCORNER_UP) == MASK_DCORNER_UP || (WK & MASK_DCORNER_DOWN) == MASK_DCORNER_DOWN) {
        score -= 40;
    }

    return score;
}

inline int centerKingScore(Position &pos) {
    const uint32_t BK = pos.K & pos.BP;
    const uint32_t WK = pos.K & pos.WP;

    const int scoreWhite = _popcnt32(WK & CENTER_KING);
    const int scoreBlack = _popcnt32(BK & CENTER_KING);

    return 10 * (scoreWhite - scoreBlack);
}
inline int centerPawnScore(Position &pos) {
    const uint32_t BP = (~pos.K) & pos.BP;
    const uint32_t WP = (~pos.K) & pos.WP;

    const int scoreWhite = _popcnt32(WP & CENTER_KING);
    const int scoreBlack = _popcnt32(BP & CENTER_KING);

    return 2 * (scoreWhite - scoreBlack);
}


#endif //CHECKERSTEST_GAMEEVALUATION_H

