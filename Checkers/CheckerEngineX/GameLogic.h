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
    uint64_t node_counter{0ull};
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

namespace Search {

    void search_root(Local &local, Line &line, Board &board, Value alpha, Value beta, Depth depth);

    void search_asp(Local &local, Line &line, Board &board, Value last_score, Depth depth);

    Value search(Board &board, Line &line, Value alpha, Value beta, Ply ply, Depth depth, Move skip_move, bool prune);

    void move_loop(Local &local, Board &board, Line &pv, MoveListe &liste);


    Value qs(Board &board, Line &pv, Value alpha, Value beta, Ply ply);

    Value searchMove(Move move, Local &local, Board &board, Line &line, int extension);

    Depth reduce(Local &local, Board &board, Move move, bool in_pv_line);

}


void setHashSize(uint32_t hash);

Value searchValue(Board &board, Value alpha, Value beta, Move &best, int depth, uint32_t time, bool print);

Value searchValue(Board &board, Move &best, int depth, uint32_t time, bool print);

Value searchValue(Board &board, int depth, uint32_t time, bool print);

void initialize();


#endif //CHECKERSTEST_GAMELOGIC_H