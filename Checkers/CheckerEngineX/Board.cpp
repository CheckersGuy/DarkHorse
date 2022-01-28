//
// Created by Robin on 09.05.2017.
//

#include "Board.h"

Position &Board::get_position() {
    return pStack[pCounter];
}

Board::Board(const Board &other) {
    std::copy(other.pStack.begin(), other.pStack.end(), pStack.begin());
    std::copy(other.moveStack.begin(), other.moveStack.end(), moveStack.begin());
    this->pCounter = other.pCounter;
}

void Board::print_board() const {
    Position pos = pStack[pCounter];
    pos.print_position();
}

//Requires some more checking here
//weird that I can not figure that out
void Board::make_move(Move move) {
    pStack[pCounter + 1] = pStack[pCounter];
    this->pCounter++;
    moveStack[pCounter] = move;
    Zobrist::update_zobrist_keys(get_position(), move);
    pStack[pCounter].make_move(move);


}

void Board::undo_move(Move move) {
    this->pCounter--;
}

Board &Board::operator=(Position pos) {
    get_position().BP = pos.BP;
    get_position().WP = pos.WP;
    get_position().K = pos.K;
    get_position().color = pos.color;
    get_position().key = Zobrist::generate_key(get_position());
    return *this;
}


Color Board::get_mover() const {
    return pStack[pCounter].get_color();
}

bool Board::is_silent_position() {
    return (get_position().get_jumpers<WHITE>() == 0u && get_position().get_jumpers<BLACK>() == 0u);
}

uint64_t Board::get_current_key() const {
    return pStack[pCounter].key;
}

bool Board::is_repetition() const {
    int counter = 0;
    const Position check = pStack[pCounter];
    for (int i = pCounter; i >= 0; i -= 2) {
        if (pStack[i] == check) {
            counter++;
        }
        if (counter >= 2) {
            return true;
        }


    }

    return false;
}

bool Board::isRepetition2() const {
    int counter = 0;
    const Position check = pStack[pCounter];
    for (int i = pCounter; i >= 0; i -= 2) {

        if (pStack[i] == check) {
            counter++;
        }
        if (counter >= 2)
            return true;

        if ((moveStack[i].to & pStack[i].K) == 0)
            return false;

        if (moveStack[i].is_capture()) {
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

