//
// Created by Robin on 09.05.2017.
//

#include "Board.h"

Position &Board::getPosition() {
    return pStack[pCounter];
}

Board::Board(const Board &board) {
    for (int i = 0; i < moves.size(); ++i) {
        this->moves[i] = board.moves[i];
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
    moves[pCounter++] = move;
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

        if ((current.getFrom() & (pStack[i].K)) == 0u)
            return false;


        if (current.isCapture() || current.isPromotion(pStack[i].K))
            return false;


    }

    return false;
}

