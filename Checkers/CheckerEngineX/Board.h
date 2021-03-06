//
// Created by Robin on 09.05.2017.
//

#ifndef CHECKERSTEST_BOARD_H
#define CHECKERSTEST_BOARD_H

#include "Move.h"
#include <string>
#include "types.h"
#include "Position.h"
#include "Zobrist.h"

class Board {

private:
    std::array<Position, MAX_PLY+400> pStack;
    std::array<Move, MAX_PLY+400> moves;

public:
    int pCounter = 0;

    Board() = default;

    Board(const Board &board);

    void printBoard() const;

    void makeMove(Move move);

    void undoMove();

    bool isSilentPosition();

    Position &getPosition();

    uint64_t getCurrentKey() const;

    bool isRepetition() const;

    Color getMover() const;

    Board &operator=(Position pos);
};


#endif //CHECKERSTEST_BOARD_H
