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

public:
    std::array<Position, MAX_PLY+600> pStack;
public:
    int last_non_rev{0};
    int pCounter = 0;

    Board() = default;

    Board(const Board &board);

    void print_board() const;

    void make_move(Move move);

    void undo_move();

    bool is_silent_position();

    Position &get_position();

    Position previous();

    uint64_t get_current_key() const;

    bool is_repetition2(int last_rev)const;

    bool is_repetition()const;

    Color get_mover() const;

    Board &operator=(Position pos);

    size_t history_length()const;

    Position history_at(size_t idx);
    //difference between make-move and play move is that
    //play move is not being used for the tree search
    void play_move(Move move);
    


};


#endif //CHECKERSTEST_BOARD_H
