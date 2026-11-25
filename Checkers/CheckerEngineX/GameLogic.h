

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
#include <array>
#include <atomic>
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

struct ThreadData; // Thread.h
extern uint64_t counter;
extern uint64_t both_counter;
extern int last_eval;
extern std::atomic<bool>
    stop_search; // set by the main thread, polled by helpers

enum NodeType {
  ROOT,
  PV,
  NONPV,
};

struct RootMove {
  Move move{};
  Value score{-SCORE_INFINITE};
  Value previous_score{-SCORE_INFINITE};
  Line pv;
};

struct Sample {
  Position pos;
  Value evaluation;
};

// Position sampling for data generation (--generate): every main-search node
// without a capture for the side to move is offered once per game (in-game
// dedup). A new position is kept with probability
// sample_prob * band_accept[band(|static eval|)] (Bernoulli), and the kept
// stream is reservoir-sampled down to max_samples per game. Defaults (p = 1,
// cap 2500, all bands 1) = the original reservoir sampling.
struct Reservoir {
  bool enabled = false;        // only --generate samples (no cost elsewhere)
  int max_samples = 2500;      // cap per game
  double sample_prob = 1.0;    // keep probability of a new position
  // |static eval| bands: <= 50, 51-200, 201-400, > 400
  std::array<double, 4> band_accept{1.0, 1.0, 1.0, 1.0};
  uint64_t seen = 0;           // positions that passed the Bernoulli filter
  uint64_t offered = 0;        // distinct positions offered this game
  std::vector<Sample> samples;
  std::mt19937_64 generator;
  std::uniform_real_distribution<double> unit{0.0, 1.0};
  std::unordered_set<Position> seen_positions;

  static int band(Value value) {
    const int a = std::abs(static_cast<int>(value));
    return a <= 50 ? 0 : (a <= 200 ? 1 : (a <= 400 ? 2 : 3));
  }

  void add_sample(Position pos, Value value) {
    if (!seen_positions.insert(pos).second) {
      return;
    }
    ++offered;
    const double p = sample_prob * band_accept[band(value)];
    if (p < 1.0 && unit(generator) >= p) {
      return;
    }
    if (samples.size() < static_cast<size_t>(max_samples)) {
      samples.emplace_back(Sample{pos, value});
    } else {
      std::uniform_int_distribution<uint64_t> distrib(0, seen);
      const auto new_index = distrib(generator);
      if (new_index < static_cast<uint64_t>(max_samples)) {
        samples[new_index] = Sample{pos, value};
      }
    }
    seen++;
  }

  void clear() {
    samples.clear();
    seen_positions.clear();
    seen = 0;
    offered = 0;
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

// Root bookkeeping of the last searchValueMultiPV call (main thread), for the
// label diagnostics of the rescoring eval loop (--label_diag)
struct RootDiag {
  int completed_depth = 0;   // last fully completed iteration
  Value completed_score = 0; // its (exact) score
  bool interrupted = false;  // stopped inside an iteration
  bool partial_used = false; // returned the interrupted iteration's best move/score
  bool partial_lower = false; // ... and that score came from a fail high (lower bound)
  int fail_lows = 0, fail_highs = 0; // aspiration fails in the interrupted iteration
  Value fail_low_score = 0; // score of the last fail low there (an upper bound)
  Value returned_score = 0;
  uint64_t nodes = 0;
};
extern RootDiag root_diag;

// --log_evals FILE --log_rate N: sample every N-th evaluated position
void set_eval_log(const std::string &path, int rate);

namespace Search {

Value search_asp(ThreadData &td, Board &board, Value last_score, Depth depth);

template <NodeType type>
Value search(ThreadData &td, bool cutnode, Board &board, Ply ply, Line &line,
             Value alpha, Value beta, Depth depth, Move excluded,
             bool is_sing_search);

template <NodeType type>
Value qs(ThreadData &td, Board &board, Ply ply, Line &pv, Value alpha,
         Value beta, Depth depth, Move excluded, bool is_sing_search);

Depth reduce(bool improving, int move_index, Depth depth, Ply ply, Board &board,
             Move move, bool in_pv, bool cutnode, float move_prob,
             bool have_prob);

} // namespace Search

std::vector<RootMove> searchValueMultiPV(Board board, int numPV, int depth,
                                         uint32_t time, size_t max_nodes,
                                         bool print, std::ostream &stream);

// The ThreadData overloads use that thread's network state (search); the
// others use a per-OS-thread default state (tools, book code, main.cpp).
int get_mlh_estimate(ThreadData &td, Position pos);
int get_mlh_estimate(Position pos);
Value evaluate(ThreadData &td, Position pos, Ply ply);
Value evaluate(Position pos, Ply ply);

std::array<float, 40> get_probability_distribution(MoveListe &liste,
                                                   Position pos);

extern Network<2048, 32, 32, 1, 128> network;

extern Network<128, 32, 32, 1, 128> mlh_net;

extern Network<1024, 32, 32, 128, 128> policy;

#endif // CHECKERSTEST_GAMELOGIC_H
