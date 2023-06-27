

#ifndef CHECKERSTEST_GAMELOGIC_H
#define CHECKERSTEST_GAMELOGIC_H

#define CHECKERBOARD

#include "Bits.h"
#include "Board.h"
#include "Line.h"
#include "MGenerator.h"
#include "Transposition.h"
#include <Network.h>
#include <algorithm>
#include <chrono>
#include <types.h>
struct SearchGlobal {
  uint32_t sel_depth;
#ifdef CHECKERBOARD
  char *reply;
#endif
  // will be called whenever we find a new move
  void new_move();

  // will be called when the evaluation changes
  void score_update();
};

extern SearchGlobal glob;

Value qsSearch(Board &board, Line &line, Ply ply, Value alpha, Value beta);

Value search(Board board, Move &best, Depth depth, uint32_t time, bool print);

namespace Search {

void search_root(Local &local, Line &line, Board &board, Value alpha,
                 Value beta, Depth depth);

void search_root(Local &local, Line &line, Board &board, Value alpha,
                 Value beta, Depth depth, std::vector<Move> &exluded_moves);

void search_asp(Local &local, Board &board, Value last_score, Depth depth);

Value search(bool in_pv, Board &board, Line &line, Value alpha, Value beta,
             Ply ply, Depth depth, int last_rev, Move previous,
             Move previous_own);

void move_loop(bool in_pv, Local &local, Board &board, Line &pv,
               MoveListe &liste, int last_rev);

Value qs(bool in_pv, Board &board, Line &pv, Value alpha, Value beta, Ply ply,
         Depth depth, int last_rev);

Value searchMove(bool in_pv, Move move, Local &local, Board &board, Line &line,
                 int extension, int last_rev);

Depth reduce(Local &local, Board &board, Move, bool in_pv);

} // namespace Search

Value searchValue(Board board, Move &best, int depth, uint32_t time, bool print,
                  std::ostream &stream);

void initialize();

void initialize(uint64_t seed);

extern Network network, network2;

#endif // CHECKERSTEST_GAMELOGIC_H
