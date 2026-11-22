#ifndef CHECKERSTEST_THREAD_H
#define CHECKERSTEST_THREAD_H

#include "Board.h"
#include "GameLogic.h"
#include "Line.h"
#include "Move.h"
#include "types.h"
#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

// Everything one search thread modifies during a search. Lazy SMP: every thread
// owns one of these; the only state shared between threads is the TT.
struct ThreadData {
  int id = 0; // 0 = main thread

  Board board;

  // Written only by the owning thread; other threads read it to sum up node
  // counts, hence atomic (relaxed load/store compiles to plain moves on x86).
  std::atomic<uint64_t> nodes{0};

  uint32_t sel_depth = 0;
  int root_depth = 0;
  int completed_depth = 0;

  std::array<Value, MAX_PLY + 10> static_evals;
  std::array<bool, MAX_PLY + 10> ttpv_stack{}; // is_tt_pv per ply

  // root bookkeeping
  Move root_partial{};
  Value root_partial_score = -SCORE_INFINITE;
  Move root_prev_best{};
  std::array<Move, 40> pv_excluded_moves;
  int num_pv_excluded = 0;
  Line main_pv;
  std::vector<RootMove> results;

  // per-thread network evaluation state; weights are shared (see Network.h)
  decltype(network)::State net_state;
  decltype(mlh_net)::State mlh_state;
  decltype(policy)::State policy_state;

#ifdef SEARCH_STATS
  // Move-ordering statistics (bench builds only, -DSEARCH_STATS).
  // A cutoff is "policy-ordered" when the quiet moves at that node were
  // sorted by the policy net; its rank is the position among those sorted
  // moves (the TT move, if tried first, is not counted in the rank).
  struct CutStats {
    uint64_t cutoffs = 0;        // all beta cutoffs in the main search
    uint64_t cut_first = 0;      // ... caused by the first move tried
    uint64_t cut_tt = 0;         // ... caused by the TT move
    uint64_t pol_cutoffs[2] = {}; // policy-ordered cutoffs [0]=no TT move, [1]=TT move failed
    uint64_t pol_rank[2][4] = {}; // rank 0,1,2,3+
    uint64_t pol_rank_sum[2] = {};
    void reset() { *this = CutStats{}; }
  } cut_stats;
#endif

  uint64_t inc_nodes() {
    const uint64_t n = nodes.load(std::memory_order_relaxed) + 1;
    nodes.store(n, std::memory_order_relaxed);
    return n;
  }
};

// The search threads; threads[0] is the main thread. Resized via
// set_num_threads(), never during a search.
extern std::vector<std::unique_ptr<ThreadData>> threads;

void set_num_threads(int n);

int get_num_threads();

ThreadData &main_td();

uint64_t total_nodes();

#endif // CHECKERSTEST_THREAD_H
