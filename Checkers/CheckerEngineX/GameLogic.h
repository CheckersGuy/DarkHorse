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
#include <Network.h>
#include <types.h>

struct SearchGlobal {
    uint32_t sel_depth;

    //will be called whenever we find a new move
    void new_move();

    // will be called when the evaluation changes
    void score_update();
};

extern SearchGlobal glob;

struct Local {
    Value alpha, beta;
    Value best_score{-INFINITE};
    Value sing_score;
    Depth depth;
    Ply ply;
    int i;
    Move skip_move;
    Move sing_move;
    Move move;
};

void use_classical(bool flag);

Value alphaBeta(Board &board, Line &line, Ply ply, Depth depth, Value alpha, Value beta, bool in_pv);

Value qsSearch(Board &board, Line &line, Ply ply, Value alpha, Value beta);

Value search(Board board, Move &best, Depth depth, uint32_t time, bool print);

namespace Search {


    void search_root(Local &local, Line &line, Board &board, Value alpha, Value beta, Depth depth);

    void search_root(Local &local, Line &line, Board &board, Value alpha, Value beta, Depth depth,
                     std::vector<Move> &exluded_moves);

    void search_asp(Local &local, Board &board, Value last_score, Depth depth);

    Value search(bool in_pv, Board &board, Line &line, Value alpha, Value beta, Ply ply, Depth depth, Move skip_move);

    void move_loop(bool in_pv, Local &local, Board &board, Line &pv, MoveListe &liste);

    Value qs(bool in_pv, Board &board, Line &pv, Value alpha, Value beta, Ply ply, Depth depth);

    Value searchMove(bool in_pv, Move move, Local &local, Board &board, Line &line, int extension);

    Depth reduce(Local &local, Board &board, Move, bool in_pv);

}

Value searchValue(Board board, Move &best, int depth, uint32_t time, bool print);

Value searchValue(Board &board, int depth, uint32_t time, bool print);

void initialize();

void initialize(uint64_t seed);

extern Network network;


#endif //CHECKERSTEST_GAMELOGIC_H