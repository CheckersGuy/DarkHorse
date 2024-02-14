

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

#ifdef _WIN32
#include "Endgame.h"
#include "egdb.h"
inline TableBase tablebase;
#endif

extern uint64_t nodeCounter;

enum NodeType {
  ROOT,
  PV,
  NONPV,
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

namespace Search {

Value search_asp(Board &board, Value last_score, Depth depth);

template <NodeType type>
Value search(Board &board, Ply ply, Line &line, Value alpha, Value beta,
             Depth depth, int last_rev, Move excluded, bool is_sing_search);
template <NodeType type>
Value qs(Board &board, Ply ply, Line &pv, Value alpha, Value beta, Depth depth,
         int last_rev, Move excluded, bool is_sing_search);

Depth reduce(int move_index, Depth depth, Board &board, Move, bool in_pv);

} // namespace Search

Value searchValue(Board &board, Move &best, int depth, uint32_t time,
                  bool print, std::ostream &stream);

int get_mlh_estimate(Position pos);

extern Network<2048, 32, 32, 1> network;
extern Network<512, 32, 32, 1> mlh_net;
extern Network<512, 32, 32, 128> policy;
#endif // CHECKERSTEST_GAMELOGIC_H
