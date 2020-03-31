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
    uintuint64_t node_counter{0u};
    Value alpha, beta;
    Value best_score;
    Line pv_line;
    Depth depth;
    Ply ply;
    int i;
    Move skip_move;
    Move move;
    MoveListe move_list;
    bool in_pv_line{false};
    bool prune{false};
    Board board;
};

namespace Search {

    template<NodeType type>
    Value search(Local &local, Value alpha, Value beta, Ply ply, Depth depth, Line &line);

    void move_loop(Local &local);

    template<NodeType type>
    Value qs(Local &local, Ply ply);

    template<NodeType type>
    void searchMove(Move move, Local &local);

    Depth reduce(Local &local, Move move);

    Depth extend(Local &local, Move move);

}


void setHashSize(uint32_t hash);

template<NodeType type>
Value quiescene(Board &board, Value alpha, Value beta, Line &pv, int ply);

template<NodeType type>
Value alphaBeta(Board &board, Value alpha, Value beta, Line &localPV, int ply, int depth, bool prune);


Value searchValue(Board &board, Move &best, int depth, uint32_t time, bool print);

Value searchValue(Board &board, int depth, uint32_t time, bool print);

void initialize();


#endif //CHECKERSTEST_GAMELOGIC_H