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
    Line pv_line;
    Depth depth;
    Ply ply;
    int i;
    Move skip_move;
    Move move;
    MoveListe move_list;
    bool prune{false};
    Board board;
};

namespace Search {

    void search_root(Local& local, Value alpha, Value beta, Depth depth);

    template<NodeType type>
    Value search(Local &local,Line& line, Value alpha, Value beta, Ply ply, Depth depth, bool prune);

    template<NodeType type>
    void move_loop(Local &local, Line& line);

    template<NodeType type>
    Value qs(Local &local, Line& line, Ply ply);

    template<NodeType type>
    void searchMove(Move move, Local &local, Line& line);

    Depth reduce(Local &local, Move move, bool in_pv_line);

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