//
// Created by Robin on 09.05.2017.
//

#include "Board.h"

Position &Board::getPosition() {
    return pStack[pCounter];
}

Board::Board(const Board &other) {
    std::copy(other.pStack.begin(), other.pStack.end(), pStack.begin());
    std::copy(other.moves.begin(), other.moves.end(), moves.begin());
    this->pCounter = other.pCounter;
}

void Board::printBoard() {
    Position pos = pStack[pCounter];
    pos.printPosition();
}

void Board::makeMove(Move move) {
    assert(!move.isEmpty());
    pStack[pCounter + 1] = pStack[pCounter];
    this->pCounter++;
    Zobrist::doUpdateZobristKey(getPosition(), move);
    pStack[pCounter].makeMove(move);
    moves[pCounter] = move;
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
    return getPosition().key;
}

bool Board::isRepetition() {
    if (pCounter < 4)
        return false;

    int counter = 0;

    for (auto i = pCounter; i > 0; i -= 2) {

        if(moves[i].isCapture() || moves[i].isPromotion(pStack[i-1].K))
            return false;

        if (pStack[i] == getPosition()) {
            counter++;
        }
        if (counter == 2)
            return true;

    }

    return false;
}

