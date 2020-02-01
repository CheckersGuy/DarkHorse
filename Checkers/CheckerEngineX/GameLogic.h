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




void setHashSize(uint32_t hash);

template<NodeType type>
Value quiescene(Board &board, Value alpha, Value beta, Line &pv, int ply);

template<NodeType type>Value alphaBeta(Board &board, Value alpha, Value beta,Line& localPV, int ply, int depth, bool prune);


Value searchValue(Board &board, Move &best, int depth, uint32_t time, bool print);

Value searchValue(Board &board, int depth, uint32_t time, bool print);

 void initialize();




#endif //CHECKERSTEST_GAMELOGIC_H