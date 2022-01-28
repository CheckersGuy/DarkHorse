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
    std::array<Position, MAX_PLY+600> pStack;
    std::array<Move,MAX_PLY+600> moveStack;

public:
    int pCounter = 0;

    Board() = default;

    Board(const Board &board);

    void print_board() const;

    void make_move(Move move);

    void undo_move(Move move);

    bool is_silent_position();

    Position &get_position();

    Position previous();

    uint64_t get_current_key() const;

    bool is_repetition() const;

    bool isRepetition2() const;

    Color get_mover() const;

    Board &operator=(Position pos);
};


#endif //CHECKERSTEST_BOARD_H
