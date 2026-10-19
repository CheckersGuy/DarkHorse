

#ifndef CHECKERSTEST_GAMELOGIC_H
#define CHECKERSTEST_GAMELOGIC_H
#include "Bits.h"
#include "Board.h"
#include "Line.h"
#include "MGenerator.h"
#include "Move.h"
#include "Network.h"
#include "Transposition.h"
#include "types.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <random>
#include <unordered_set>
#ifdef _WIN32
#include "Endgame.h"
#include "egdb.h"
inline TableBase tablebase;
#endif

extern uint64_t nodeCounter;
extern uint64_t counter;
extern uint64_t both_counter;
extern int last_eval;
extern bool stop_search;
enum NodeType {
  ROOT,
  PV,
  NONPV,
};

struct RootMove {
  Move move{};
  Value score{-INFINITE};
  Value previous_score{-INFINITE};
  Line pv;
};

struct Sample {
  Position pos;
  Value evaluation;
};

struct Reservoir {
  constexpr static int max_samples = 2500;
  uint64_t seen = 0;
  std::vector<Sample> samples;
  std::mt19937_64 generator;
  std::unordered_set<Position> seen_positions;

  void add_sample(Position pos, Value value) {
    // filling the entire reservoir first

    if (seen_positions.contains(pos)) {
      return;
    }

    if (samples.size() < max_samples) {
      samples.emplace_back(Sample{pos, value});
    } else {
      std::uniform_int_distribution<int> distrib(0, seen);
      const auto new_index = distrib(generator);
      if (new_index < max_samples) {
        samples[new_index] = Sample{pos, value};
      }
    }
    seen_positions.insert(pos);
    seen++;
  }

  void clear() {
    samples.clear();
    seen_positions.clear();
    seen = 0;
  }
};

extern Reservoir reservoir;
struct SearchGlobal {
  uint32_t sel_depth;
#ifdef CHECKERBOARD
  char *reply;
  int *playnow;
#endif

  // will be called whenever we find a new move
  void new_move();

  // will be called when the evaluation changes
  void score_update();
};

extern SearchGlobal glob;

namespace Search {

Value search_asp(Board &board, Value last_score, Depth depth);

template <NodeType type>
Value search(bool cutnode, Board &board, Ply ply, Line &line, Value alpha,
             Value beta, Depth depth, Move excluded, bool is_sing_search);

template <NodeType type>
Value qs(Board &board, Ply ply, Line &pv, Value alpha, Value beta, Depth depth,
         Move excluded, bool is_sing_search);

Depth reduce(bool improving, int move_index, Depth depth, Ply ply, Board &board,
             Move move, bool in_pv, bool cutnode);

} // namespace Search

std::vector<RootMove> searchValueMultiPV(Board board, int numPV, int depth,
                                         uint32_t time, size_t max_nodes,
                                         bool print, std::ostream &stream);

int get_mlh_estimate(Position pos);
Value evaluate(Position pos, Ply ply);

std::array<float, 40> get_probability_distribution(MoveListe &liste,
                                                   Position pos);

extern Network<2048, 32, 32, 1, 128> network;

extern Network<128, 32, 32, 1, 12> mlh_net;

extern Network<1024, 32, 32, 128, 12> policy;

#endif // CHECKERSTEST_GAMELOGIC_H
