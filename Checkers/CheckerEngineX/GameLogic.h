

#ifndef CHECKERSTEST_GAMELOGIC_H
#define CHECKERSTEST_GAMELOGIC_H

#include "Bits.h"
#include "Board.h"
#include "Line.h"
#include "MGenerator.h"
#include "Move.h"
#include "MovePicker.h"
#include "Network.h"
#include "Transposition.h"
#include "types.h"
#include <Network.h>
#include <algorithm>
#include <chrono>
#include <types.h>

enum NodeType {
  ROOT,
  PV,
  NONPV,
};

// Searchstack
struct Stack {
  Ply ply;
  Move previous;
  Color us;
  bool tt_hit;
};

struct SearchGlobal {
  uint32_t sel_depth;
#ifdef CHECKERBOARD
  char *reply;
#endif

  bool stop_search = false;
  // will be called whenever we find a new move
  void new_move();

  // will be called when the evaluation changes
  void score_update();
};

extern SearchGlobal glob;

Value qsSearch(Board &board, Line &line, Ply ply, Value alpha, Value beta);

Value search(Board board, Move &best, Depth depth, uint32_t time, bool print);

namespace Search {

Value search_asp(Board &board, Value last_score, Depth depth);

template <NodeType type>
Value search(Board &board, Line &line, Value alpha, Value beta, Ply ply,
             Depth depth, int last_rev);
template <NodeType type>
Value qs(Board &board, Line &pv, Value alpha, Value beta, Ply ply, Depth depth,
         int last_rev);

Depth reduce(int move_index, Depth depth, Board &board, Move, bool in_pv);

} // namespace Search

Value searchValue(Board board, Move &best, int depth, uint32_t time, bool print,
                  std::ostream &stream);

extern Network network;
#endif // CHECKERSTEST_GAMELOGIC_H
