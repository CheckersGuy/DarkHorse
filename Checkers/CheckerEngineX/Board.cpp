//
// Created by Robin on 09.05.2017.
//

#include "Board.h"
#include "BoardFactory.h"

const uint32_t S[32] = {3, 2, 1, 0, 7, 6, 5, 4, 11, 10, 9, 8, 15, 14, 13, 12, 19, 18, 17, 16, 23, 22, 21, 20, 27, 26,
                        25, 24, 31, 30, 29, 28};

Position *Board::getPosition() {
    return &pStack[pCounter];
}

Board::Board(const Board &board) {
    //Copy-Constructor goes here
    for (int i = 0; i < MAX_MOVE + MAX_PLY; ++i) {
        this->history[i] = board.history[i];
        this->pStack[i] = board.pStack[i];
    }
    this->pCounter = board.pCounter;
    this->moveCount = board.moveCount;
}

void Board::printBoard() {
    Position pos = pStack[pCounter];
    std::cout << "Color: " << ((getMover() == BLACK) ? "BLACK" : "WHITE") << std::endl;
    std::cout << "MoveCount: " << moveCount << std::endl;
    pos.printPosition();
}

void Board::makeMove(Move move) {
    assert(!move.isEmpty());
    pStack[pCounter + 1] = pStack[pCounter];
    history[pCounter] = move;
    this->pCounter++;

    pStack[pCounter].makeMove(move);
    Zobrist::doUpdateZobristKey(pStack[pCounter], move);
    this->moveCount++;
    assert(moveCount <= MAX_MOVE);

}

void Board::undoMove() {
    this->pCounter--;
    this->moveCount--;
}

