//
// Created by Robin on 09.05.2017.
//

#include "Board.h"

Position &Board::getPosition() {
    return pStack[pCounter];
}

Board::Board(const Board &other) {
    std::copy(other.moves.begin(),other.moves.end(),moves.begin());
    std::copy(other.pStack.begin(),other.pStack.end(),pStack.begin());
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
    for (auto i = pCounter - 2; i > 0; i -= 2) {
        Move &current = moves[i];
        if (pStack[i] == getPosition()) {
            return true;
        }
        if (current.isCapture() || current.isPromotion(pStack[i - 1].K))
            return false;

    }

    return false;
}

