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

constexpr uint32_t S[32] = {3u, 2u, 1u, 0u, 7u, 6u, 5u, 4u, 11u, 10u, 9u, 8u, 15u, 14u, 13u, 12u, 19u, 18u, 17u, 16u, 23u, 22u, 21u, 20u, 27u, 26u,
                            25u, 24u, 31u, 30u, 29u, 28u};


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

    Position& getPosition();

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
    return (wjumpers == 0u && bjumpers == 00);
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
