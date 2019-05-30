//
// Created by Robin on 09.05.2017.
//

#ifndef CHECKERSTEST_BOARD_H
#define CHECKERSTEST_BOARD_H

#include <iostream>
#include<string>
#include "Move.h"
#include<random>
#include "immintrin.h"
#include <cassert>
#include <string>
#include "types.h"
#include "Position.h"
#include "Zobrist.h"

constexpr uint32_t S[32] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 27, 26,
                            25, 24, 31, 30, 29, 28};


class Board {

public:
    Position pStack[MAX_PLY + MAX_MOVE];
    Move history[MAX_PLY + MAX_MOVE];
    int pCounter = 0;

    Board() = default;

    Board(const Board &board);

    void printBoard();

    void makeMove(Move move);

    void undoMove();

    bool isSilentPosition();

    bool hasJumps();

    Position *getPosition();

    uint64_t getCurrentKey();

    bool isRepetition();

    Color getMover();

    Board& operator=(const Position pos);
};


inline Color Board::getMover() {
    return pStack[pCounter].color;
}

inline bool Board::isSilentPosition() {
    const uint32_t wjumpers = pStack[pCounter].getJumpers<WHITE>();
    const uint32_t bjumpers = pStack[pCounter].getJumpers<BLACK>();
    return (wjumpers == 0 && bjumpers == 0);
}

inline bool Board::hasJumps() {
    return !isSilentPosition();
}

inline uint64_t Board::getCurrentKey() {
    return this->pStack[pCounter].key;
}

inline bool Board::isRepetition() {
    //checking for repetitions
    for (int i = pCounter - 2; i >= 0; i -= 2) {
        if (getCurrentKey() == pStack[i].key) {
            return true;
        }
        if (history[i].getPieceType() == 0 || history[i].isCapture() || history[i].isPromotion()) {
            return false;
        }

    }

    return false;
}

#endif //CHECKERSTEST_BOARD_H
