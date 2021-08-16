//
// Created by Robin on 09.05.2017.
//

#include "Board.h"

Position &Board::getPosition() {
    return pStack[pCounter];
}

Board::Board(const Board &other) {
    std::copy(other.pStack.begin(), other.pStack.end(), pStack.begin());
    this->pCounter = other.pCounter;
}

void Board::printBoard() const {
    Position pos = pStack[pCounter];
    pos.printPosition();
}

void Board::makeMove(Move move) {

    pStack[pCounter + 1] = pStack[pCounter];
    this->pCounter++;
    Zobrist::doUpdateZobristKey(getPosition(), move);
    pStack[pCounter].makeMove(move);

    if(move.isCapture()){
        pre_rev_mov_counter = rev_mov_counter;
        rev_mov_counter = pCounter;
    }


}

void Board::undoMove() {
    this->pCounter--;
    if(pCounter<=rev_mov_counter){
        rev_mov_counter = pre_rev_mov_counter;
    }
}

Board &Board::operator=(Position pos) {
    getPosition().BP = pos.BP;
    getPosition().WP = pos.WP;
    getPosition().K = pos.K;
    getPosition().color = pos.color;
    getPosition().key = Zobrist::generateKey(getPosition());
    return *this;
}


Color Board::getMover() const {
    return pStack[pCounter].getColor();
}

bool Board::isSilentPosition() {
    return (getPosition().getJumpers<WHITE>() == 0u && getPosition().getJumpers<BLACK>() == 0u);
}

uint64_t Board::getCurrentKey() const {
    return pStack[pCounter].key;
}

bool Board::isRepetition() const {
    if (pCounter < 4)
        return false;

    int counter = 0;
    const Position check = pStack[pCounter];
    for (int i = pCounter; i >= 0; i -= 2) {
        if (pStack[i] == check) {
            counter++;
        }
        if (counter >= 2)
            return true;

    }

    return false;
}

bool Board::isRepetition2() const {
    if (pCounter < 4)
        return false;

    int counter = 0;
    const Position check = pStack[pCounter];
    for (int i = pCounter; i >= rev_mov_counter; i -= 2) {

        if (pStack[i] == check) {
            counter++;
        }
        if (counter >= 2)
            return true;

    }

    return false;
}

Position Board::previous() {
    if (pCounter > 0) {
        return pStack[pCounter - 1];
    }
    return Position{};
}

