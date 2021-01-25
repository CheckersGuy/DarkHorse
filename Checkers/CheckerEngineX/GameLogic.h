//
// Created by Robin on 10.06.2017.
//

#ifndef CHECKERSTEST_GAMELOGIC_H
#define CHECKERSTEST_GAMELOGIC_H


#include "Board.h"
#include "MGenerator.h"
#include <chrono>
#include "Transposition.h"
#include "Weights.h"
#include "Line.h"
#include <algorithm>
#include "Bits.h"
#include <iterator>

struct Local {
    Value alpha, beta;
    Value best_score;
    Value sing_score;
    Depth depth;
    Ply ply;
    int i;
    Move skip_move;
    Move sing_move;
    Move move;
    bool prune{false};
    bool pv_node;
};

struct SearchGlobal{
    uint32_t sel_depth;
    //will be called whenever we find a new move
    void new_move();

    // will be called when the evaluation changes
    void score_update();
};

namespace Search {

    void search_root(Local &local, Line &line, Board &board, Value alpha, Value beta, Depth depth);

    void search_asp(Local &local, Board &board, Value last_score, Depth depth);

    Value search(Board &board, Line &line, Value alpha, Value beta, Ply ply, Depth depth, Move skip_move, bool prune);

    void move_loop(Local &local, Board &board, Line &pv, MoveListe &liste);

    Value qs(Board &board, Line &pv, Value alpha, Value beta, Ply ply,Depth depth);

    Value searchMove(Move move, Local &local, Board &board, Line &line, int extension);

    Depth reduce(Local &local, Board &board, Move move);

}


void setHashSize(uint32_t hash);

Value searchValue(Board board, Move &best, int depth, uint32_t time, bool print);

Value searchValue(Board board, int depth, uint32_t time, bool print);

void initialize();


#endif //CHECKERSTEST_GAMELOGIC_H