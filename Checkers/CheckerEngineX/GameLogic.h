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

#if defined(__GNUC__)
#define MAKRO extern"C" __attribute__((visibility("default")))
#else
#define MAKRO extern "C" __declspec(dll_export)
#endif





MAKRO void setHashSize(uint32_t hash);

bool isPVLine(Value alpha, Value beta);

template<NodeType type>Value quiescene(Board &board, Value alpha, Value beta,Line& pv, int ply) ;

template<NodeType type>Value alphaBeta(Board &board, Value alpha, Value beta,Line& localPV, int ply, int depth, bool prune);


MAKRO Value searchValue(Board &board, Move &best, int depth, uint32_t time, bool print);

Value searchValue(Board &board, int depth, uint32_t time, bool print);


MAKRO void initialize();




#endif //CHECKERSTEST_GAMELOGIC_H