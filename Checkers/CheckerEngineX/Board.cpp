//
// Created by Robin on 09.05.2017.
//

#include "Board.h"
#include "BoardFactory.h"


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
}

void Board::printBoard() {
    Position pos = pStack[pCounter];
    pos.printPosition();
}

void Board::makeMove(Move move) {
    assert(!move.isEmpty());
    pStack[pCounter + 1] = pStack[pCounter];
    history[pCounter] = move;
    this->pCounter++;

    pStack[pCounter].makeMove(move);
    Zobrist::doUpdateZobristKey(pStack[pCounter], move);
    assert(moveCount <= MAX_MOVE);

}

void Board::undoMove() {
    this->pCounter--;
}

