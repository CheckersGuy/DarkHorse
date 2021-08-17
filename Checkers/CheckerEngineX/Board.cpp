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

//Requires some more checking here
//weird that I can not figure that out
void Board::makeMove(Move move) {
    pStack[pCounter + 1] = pStack[pCounter];
    this->pCounter++;
    moveStack[pCounter]=move;
    Zobrist::doUpdateZobristKey(getPosition(), move);
    pStack[pCounter].makeMove(move);




}

void Board::undoMove(Move move) {
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
    if(rev_mov_counter>pCounter)
        std::cerr<<"Error"<<std::endl;

    int counter = 0;
    const Position check = pStack[pCounter];
    for (int i = pCounter; i >= 0; i -= 2) {
        if (pStack[i] == check) {
            counter++;
        }
        if (counter >= 2){
            return true;
        }


    }

    return false;
}

bool Board::isRepetition2() const {
    int counter = 0;
    const Position check = pStack[pCounter];
    for (int i = pCounter; i >=0; i -= 2) {

        if (pStack[i] == check) {
            counter++;
        }
        if (counter >= 2)
            return true;

        if((moveStack[i].to& pStack[i].K)==0)
            return false;

        if(moveStack[i].isCapture()){
            return false;
        }
    }

    return false;
}

Position Board::previous() {
    if (pCounter > 0) {
        return pStack[pCounter - 1];
    }
    return Position{};
}

