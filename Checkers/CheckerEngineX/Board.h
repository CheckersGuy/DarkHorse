//
// Created by Robin on 09.05.2017.
//

#ifndef CHECKERSTEST_BOARD_H
#define CHECKERSTEST_BOARD_H

#include <iostream>
#include<string>
#include "Move.h"
#include<random>
#include <cassert>
#include <string>
#include "types.h"
#include "Position.h"
#include "Zobrist.h"

class Board {

public:
    std::array<Position,MAX_PLY + MAX_MOVE>pStack;
    std::array<Move,MAX_PLY + MAX_MOVE> history;
    uint32_t pCounter = 0;

    Board() = default;

    Board(const Board& board);

    void printBoard();

    void makeMove(Move move);

    void undoMove();

    bool isSilentPosition();

    bool hasJumps();

    Position& getPosition();

    uint64_t getCurrentKey();

    bool isRepetition();

    Color getMover();

    Board& operator=(Position pos);
};


#endif //CHECKERSTEST_BOARD_H
