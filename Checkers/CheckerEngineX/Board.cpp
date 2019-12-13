//
// Created by Robin on 09.05.2017.
//

#include "Board.h"

Position &Board::getPosition() {
    return pStack[pCounter];
}

Board::Board(const Board &board) {
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
    Zobrist::doUpdateZobristKey(pStack[pCounter], move);
    pStack[pCounter].makeMove(move);

}

void Board::undoMove() {
    this->pCounter--;
}

Board &Board::operator=(Position pos) {
    getPosition().BP = pos.BP;
    getPosition().WP = pos.WP;
    getPosition().K = pos.K;
    getPosition().color = pos.color;
    getPosition().key = Zobrist::generateKey(getPosition());
    return *this;
}


Color Board::getMover() {
    return getPosition().getColor();
}

bool Board::isSilentPosition() {
    return (getPosition().getJumpers<WHITE>() == 0u && getPosition().getJumpers<BLACK>() == 0u);
}

bool Board::hasJumps() {
    return !isSilentPosition();
}

uint64_t Board::getCurrentKey() {
    return this->pStack[pCounter].key;
}

bool Board::isRepetition() {
    //checking for repetitions
    for (int i = pCounter - 2; i >= 0; i -= 2) {
        if (getPosition().key == pStack[i].key) {
            return true;
        }
        if (((history[i].getFrom() & pStack[i].K) == 0) || history[i].isCapture() || history[i].isPromotion()) {
            return false;
        }
    }

    return false;
}

