#include "GameLogic.h"
#include "Bits.h"
#include "MGenerator.h"
#include "Network.h"
#include "Position.h"
#include "Thread.h"
#include "types.h"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <unordered_map>
#include <utility>

uint64_t endTime = 1000000000;
size_t max_nodes_search = 18446744073709551615ull;
Value last_eval = -SCORE_INFINITE;
uint64_t total_counter = 0;
uint64_t diff_counter = 0;
const bool forced_cap_enabled = true;
uint64_t counter = 0;
uint64_t both_counter = 0;
uint64_t search_start_time = 0ull;
uint64_t search_time_ms = 0ull; // time budget of the current search
Value glob_current_score = -SCORE_INFINITE;
uint64_t last_nodes_per_second = 0ull;
SearchGlobal glob;

Network<2048, 32, 32, 1, 128> network;

Network<128, 32, 32, 1, 128> mlh_net;

Network<1024, 32, 32, 128, 128> policy;

Reservoir reservoir;

#ifdef _WIN32
void egdb_message_callback(char *msg) {
  if (msg == nullptr)
    return;
#ifdef CHECKERBOARD
  if (glob.reply != nullptr) {
    snprintf(glob.reply, 1024, "%s", msg);
  }
#endif
  // std::cout << msg << std::endl;
}
#endif

inline bool creates_shot_rt(const Position &pos, const Move &m) {
  return pos.color == BLACK ? pos.creates_shot<BLACK>(m.from, m.to)
                            : pos.creates_shot<WHITE>(m.from, m.to);
}

uint64_t current_nps() {
  const double elapsed_sec =
      std::max(1e-9, (double)(getSystemTime() - search_start_time) / 1e9);
  return static_cast<uint64_t>(total_nodes() / elapsed_sec);
}

void SearchGlobal::new_move() {
#ifdef CHECKERBOARD
  if (reply == nullptr)
    return;
  last_nodes_per_second = current_nps();
  snprintf(reply, 1024, "depth %d/%d score %d nodes %llu KN's %llu PV: %s",
           main_td().root_depth, main_td().sel_depth, glob_current_score,
           (unsigned long long)total_nodes(),
           (unsigned long long)(last_nodes_per_second / 1000),
           main_td().main_pv.toString(20).c_str());
#endif
}

void SearchGlobal::score_update() {
#ifdef CHECKERBOARD
  if (reply == nullptr)
    return;
  last_nodes_per_second = current_nps();
  snprintf(reply, 1024, "depth %d/%d score %d nodes %llu KN's %llu PV: %s",
           main_td().root_depth, main_td().sel_depth, glob_current_score,
           (unsigned long long)total_nodes(),
           (unsigned long long)(last_nodes_per_second / 1000),
           main_td().main_pv.toString(20).c_str());
#endif
}

static int mlh_from_raw(int out) {
  auto scaled = static_cast<float>(out) / 127.0;
  scaled = std::clamp((int)std::round(scaled * 300), 0, 300);
  return scaled;
}

int get_mlh_estimate(ThreadData &td, Position pos) {
  return mlh_from_raw(mlh_net.evaluate(td.mlh_state, pos, 0, 0));
}

int get_mlh_estimate(Position pos) {
  return mlh_from_raw(mlh_net.evaluate(pos, 0, 0));
}

std::array<float, 40> get_probability_distribution(MoveListe &liste,
                                                   Position pos) {

  std::array<float, 40> logits;
  int32_t *out = policy.default_state().out_buffer;
  float sum = 0.0f;
  for (int i = 0; i < liste.length(); ++i) {
    if (liste[i].is_capture())
      continue; // captures already scored separately in oracle
    Move scored = (pos.color == BLACK) ? liste[i].flipped() : liste[i];
    const auto index = scored.get_move_encoding();
    logits[i] = out[index] / 127.0f * POLICY_SCALE;
    logits[i] = std::exp(logits[i]);
    sum += logits[i];
  }
  for (int i = 0; i < liste.length(); ++i) {
    logits[i] /= sum;
  }
  return logits;
}

template <typename MlhFn>
Value blend_mlh(Value eval, Position pos, MlhFn mlh_fn) {
  const Value abs_eval = std::abs(eval);
  const Value ramp_lo = MLH_RAMP_LO, ramp_hi = MAX_EVAL;

  if (abs_eval <= ramp_lo) {
    return eval;
  }

  const float t = std::clamp(
      static_cast<float>(abs_eval - ramp_lo) / (ramp_hi - ramp_lo), 0.0f, 1.0f);

  const int mlh = mlh_fn(pos);

  const Value bonus = static_cast<Value>(t * std::max(0, MLH_BONUS - mlh));

  return eval + ((eval >= 0) ? bonus : -bonus);
}

inline Value value_to_tt(Value v, int ply, Position pos) {

  if (!isEval(v)) {
    return v;
  }

  return v >= TB_WIN_MAX_PLY ? v + ply : v <= TB_LOSS_MAX_PLY ? v - ply : v;
}

inline Value value_from_tt(Value v, int ply, Position pos) {
  if (!isEval(v)) {
    return v;
  }

  return v >= TB_WIN_MAX_PLY ? v - ply : v <= TB_LOSS_MAX_PLY ? v + ply : v;
}

template <typename NetFn, typename MlhFn>
static Value evaluate_impl(Position pos, Ply ply, NetFn net_fn, MlhFn mlh_fn) {
  const auto piece_count = pos.piece_count();
  if (pos.BP == 0 && pos.color == BLACK) {
    return loss(ply);
  }

  if (pos.WP == 0 && pos.color == WHITE) {
    return loss(ply);
  }

  Value eval;
#ifdef _WIN32
  auto result = tablebase.probe(pos);

  // should probably add the mlh estimate here as well

  if (result != TB_RESULT::UNKNOWN) {

    auto dtw = tablebase.probe_dtw(pos);

    if (dtw.has_value()) {
      auto actual_dtw = ply + dtw.value();
      return (result == TB_RESULT::WIN) ? -loss(actual_dtw) : loss(actual_dtw);
    }

    const auto mlh = mlh_fn(pos);
    auto tb_value = (result == TB_RESULT::WIN)    ? -tbloss(ply) + (300 - mlh)
                    : (result == TB_RESULT::LOSS) ? tbloss(ply) - (300 - mlh)
                                                  : 0;
    return tb_value;
  }
#endif

  eval = net_fn(pos, ply);
  eval = std::clamp(eval, -MAX_EVAL, MAX_EVAL);
  eval = blend_mlh(eval, pos, mlh_fn);

  return eval;
}

Value evaluate(ThreadData &td, Position pos, Ply ply) {
  return evaluate_impl(
      pos, ply,
      [&](Position p, Ply pl) {
        return network.evaluate(td.net_state, p, pl, 0);
      },
      [&](Position p) { return get_mlh_estimate(td, p); });
}

Value evaluate(Position pos, Ply ply) {
  return evaluate_impl(
      pos, ply, [](Position p, Ply pl) { return network.evaluate(p, pl, 0); },
      [](Position p) { return get_mlh_estimate(p); });
}

std::atomic<bool> stop_search{false};

[[noreturn]] static void stop_and_throw(const char *why) {
  stop_search.store(true, std::memory_order_relaxed);
  throw std::string{why};
}

static inline void check_stop(ThreadData &td, uint64_t node_count) {
  if (td.id == 0) {
    if ((node_count & 511u) == 0u && getSystemTime() >= endTime) {
      stop_and_throw("Time_out");
    }
#ifdef CHECKERBOARD
    if ((*glob.playnow) != 0) {
      stop_and_throw("Checkerboard requested stop_search");
    }
#endif
    if (node_count >= max_nodes_search) {
      stop_and_throw("Time_out");
    }
  } else if (stop_search.load(std::memory_order_relaxed)) {
    throw std::string{"Stopped"};
  }
}

static constexpr int SkipSize[20] = {1, 1, 2, 2, 2, 2, 3, 3, 3, 3,
                                     3, 3, 4, 4, 4, 4, 4, 4, 4, 4};
static constexpr int SkipPhase[20] = {0, 1, 0, 1, 2, 3, 0, 1, 2, 3,
                                      4, 5, 0, 1, 2, 3, 4, 5, 6, 7};

static void reset_thread(ThreadData &td, const Board &root_board,
                         int actualPV) {
  td.board = root_board;
  td.sel_depth = 0u;
  td.nodes.store(0, std::memory_order_relaxed);
  td.main_pv.clear();
  td.num_pv_excluded = 0;
  td.completed_depth = 0;
  td.results.assign(actualPV, RootMove{});
  td.ext_acc.fill(0);
#ifdef SEARCH_STATS
  td.cut_stats.reset();
#endif
}

static void iterative_deepening(ThreadData &td, int actualPV, int depth,
                                bool print, std::ostream &stream) {
  Board &board = td.board;
  std::vector<RootMove> &results = td.results;
  const bool is_main = (td.id == 0);
  size_t total_time = 0;
  Move tm_prev_best{};
  int tm_stable = 0; // consecutive iterations with the same best move
  double tm_iter_ms = 0.0, tm_prev_iter_ms = 0.0; // last two iteration times
  double tm_iter_start_ms = 0.0;

  for (int d = 1; d <= depth; ++d) {
    if (!is_main) {
      if (stop_search.load(std::memory_order_relaxed))
        break;
      const int i = (td.id - 1) % 20;
      if (((d + SkipPhase[i]) / SkipSize[i]) % 2 != 0)
        continue;
    }
    bool stopped = false;
    td.num_pv_excluded = 0;
    if (is_main)
      tm_iter_start_ms = (double)(getSystemTime() - search_start_time) / 1e6;

    for (int pv_idx = 0; pv_idx < actualPV; ++pv_idx) {
      td.root_partial = Move{};
      td.root_partial_score = -SCORE_INFINITE;
      network.refresh(td.net_state);
      mlh_net.refresh(td.mlh_state);
      policy.refresh(td.policy_state);

      for (auto k = 0; k < td.static_evals.size(); ++k) {
        td.static_evals[k] = -SCORE_INFINITE;
      }

      auto start_time = std::chrono::high_resolution_clock::now();
      try {
        td.root_prev_best = results[pv_idx].move;
        td.root_depth = d;
        Value score =
            Search::search_asp(td, board, results[pv_idx].previous_score, d);
        results[pv_idx].score = score;
        results[pv_idx].pv = td.main_pv;
        results[pv_idx].move =
            (td.main_pv.length() > 0) ? td.main_pv[0] : Move{};
      } catch (std::string &msg) {

        if (pv_idx == 0 && actualPV == 1 && !td.root_partial.is_empty()) {
          results[0].move = td.root_partial;
          results[0].score =
              td.root_partial_score; // fail-high scores are lower bounds
          results[0].pv.clear();
          results[0].pv.addMove(td.root_partial);
        }

        stopped = true;
        break;
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

      if (!results[pv_idx].move.is_empty()) {
        td.pv_excluded_moves[td.num_pv_excluded++] = results[pv_idx].move;
      }
    }

    if (stopped)
      break;

    for (auto &rm : results)
      rm.previous_score = rm.score;

    // sort-by-score happens here, strictly before printing
    std::sort(
        results.begin(), results.end(),
        [](const RootMove &a, const RootMove &b) { return a.score > b.score; });
    td.completed_depth = d;

    if (!is_main)
      continue;

    glob_current_score = results[0].score;
    glob.score_update();

    if (print) {
      double time_seconds = (double)total_time / 1000.0;
      const uint64_t nps = current_nps();
      for (int k = 0; k < actualPV; ++k) {
        stream << "info depth " << d << " multipv " << (k + 1) << " score "
               << results[k].score << " time " << time_seconds << " nodes "
               << total_nodes() << " nps " << (nps / 1000) << "KN's pv "
               << results[k].pv.toString(15) << "\n";
      }
      std::cout << std::endl;
    }

    // Time management: don't start an iteration that probably cannot finish,
    // and stop earlier when the best move has been stable for a while.
    if (TM_ENABLED) {
      const Move best = results[0].move;
      tm_stable = (d > 1 && best == tm_prev_best) ? tm_stable + 1 : 0;
      tm_prev_best = best;

      const double elapsed_ms =
          (double)(getSystemTime() - search_start_time) / 1e6;
      tm_prev_iter_ms = tm_iter_ms;
      tm_iter_ms = elapsed_ms - tm_iter_start_ms;

      if (d >= TM_MIN_DEPTH) {
        const double growth =
            std::clamp(tm_prev_iter_ms > 0.0 ? tm_iter_ms / tm_prev_iter_ms
                                             : TM_GROWTH_MAX,
                       TM_GROWTH_MIN, TM_GROWTH_MAX);
        const double budget =
            (double)search_time_ms * TM_STABILITY[std::min(tm_stable, 4)];
        if (elapsed_ms + tm_iter_ms * growth > budget)
          break;
      }
    }
  }
}

std::vector<RootMove> searchValueMultiPV(Board root_board, int numPV, int depth,
                                         uint32_t time, size_t max_nodes,
                                         bool print, std::ostream &stream) {
  ThreadData &td = main_td();
  td.board = root_board;
  Board &board = td.board;

#ifdef CHECKERBOARD
  tablebase.reply = glob.reply;
#endif
  max_nodes_search = max_nodes;
  td.sel_depth = 0u;
  TT.age_counter = (TT.age_counter + 1) & 63ull;
  stop_search.store(false, std::memory_order_relaxed);
  for (auto &t : threads)
    t->nodes.store(0, std::memory_order_relaxed);
  td.main_pv.clear();

  MoveListe rootList;
  get_moves(board.get_position(), rootList);
  if (rootList.length() == 1) {
    RootMove root;
    Line line;
    root.score = Search::qs<ROOT>(td, board, 0, line, -EVAL_INFINITE,
                                  EVAL_INFINITE, depth, Move{}, false);
    ;
    root.move = rootList[0];
    return {root};
  }

  const bool forced = rootList.length() == 1;
  uint32_t eff_time = time;
  int eff_depth = depth;
  if (forced && forced_cap_enabled) { // false in eval-loop/book so those get a
    eff_time = std::min<uint32_t>(time, 1); // ms
    eff_depth = std::min(depth, 1);
  }

  const int actualPV = std::min(numPV, rootList.length());
  if (actualPV <= 0) {
    return {};
  }

  endTime = getSystemTime() + static_cast<uint64_t>(time) * 1000000;
  search_start_time = getSystemTime();
  search_time_ms = time;

  for (auto &t : threads)
    reset_thread(*t, root_board, actualPV);

  // Lazy SMP: helpers run the same search on their own board copies and only
  // share the TT; the main thread's result is the one that gets played.
  std::vector<std::thread> helpers;
  for (size_t i = 1; i < threads.size(); ++i) {
    helpers.emplace_back([&, i] {
      iterative_deepening(*threads[i], actualPV, depth, false, stream);
    });
  }

  iterative_deepening(td, actualPV, depth, print, stream);

  stop_search.store(true, std::memory_order_relaxed);
  for (auto &h : helpers)
    h.join();

  std::vector<RootMove> &results = td.results;
  last_eval = results[0].score;

#ifdef SEARCH_STATS
  if (print) {
    ThreadData::CutStats sum{};
    for (const auto &t : threads) {
      const auto &c = t->cut_stats;
      sum.cutoffs += c.cutoffs;
      sum.cut_first += c.cut_first;
      sum.cut_tt += c.cut_tt;
      for (int k = 0; k < 2; ++k) {
        sum.pol_cutoffs[k] += c.pol_cutoffs[k];
        sum.pol_rank_sum[k] += c.pol_rank_sum[k];
        for (int r = 0; r < 4; ++r)
          sum.pol_rank[k][r] += c.pol_rank[k][r];
      }
    }
    stream << "info stats cutoffs " << sum.cutoffs << " first " << sum.cut_first
           << " tt " << sum.cut_tt;
    const char *name[2] = {"nott", "ttfail"};
    for (int k = 0; k < 2; ++k) {
      stream << " " << name[k] << " " << sum.pol_cutoffs[k] << " ranksum "
             << sum.pol_rank_sum[k] << " r";
      for (int r = 0; r < 4; ++r)
        stream << " " << sum.pol_rank[k][r];
    }
    stream << std::endl;
  }
#endif

  td.num_pv_excluded = 0;
  return results;
}

namespace Search {
Depth reduce(bool improving, int move_index, Depth depth, Ply ply, Board &board,
             Move move, bool in_pv, bool cutnode, float move_prob,
             bool have_prob) {

  if (move_index >= 1 && depth >= 2 && !move.is_capture()) {
    const int d_idx = std::min(depth - 1, LMR_MAX_DEPTH - 1);
    int m_idx = std::min(move_index, LMR_MAX_MOVE_INDEX - 1);

    m_idx =
        have_prob
            ? (std::clamp((int)std::lround(1.0f / std::max(move_prob, LMR_PROB_FLOOR_X1000 / 1000.0f)),
                          1, LMR_MAX_MOVE_INDEX - 1) -
               1)
            : m_idx;
    // fixed point (1/1024 ply); converted to plies by the caller
    int red = LMR_TABLE_2D[d_idx][m_idx];

    if (in_pv) {
      red = std::max(0, red - LMR_PV_FP);
    }
    red += !improving ? LMR_NOT_IMPROVING_FP : 0;
    return red;
  }
  return 0;
}

// whole plies of a fractional depth (1/1024 ply), rounded to the nearest ply;
// the remainder (|rem| <= 1/2 ply) is carried on, so small fractions neither
// cost nor cancel a whole ply but accumulate along the line
static inline int fp_whole(int fp) {
  return fp >= 0 ? (fp + LMR_ONE / 2) / LMR_ONE
                 : -((-fp + LMR_ONE / 2) / LMR_ONE);
}

template <NodeType type>
Value search(ThreadData &td, bool cutnode, Board &board, Ply ply, Line &pv,
             Value alpha, Value beta, Depth depth, Move excluded,
             bool is_sing_search) {

  constexpr bool is_root = (type == ROOT);
  constexpr bool in_pv = (type == ROOT) || (type == PV);
  constexpr NodeType next_type = (type == ROOT) ? PV : type;
  pv.clear();
  const uint64_t node_count = td.inc_nodes();
  check_stop(td, node_count);
  // fractional depth carried in from the parent (1/1024 ply); reductions of
  // this node (alpha raise) adjust it, children receive the remainder
  int node_acc = td.ext_acc[ply];

  if (depth <= 0) {
    return Search::qs<next_type>(td, board, ply, pv, alpha, beta, depth, Move{},
                                 is_sing_search);
  }
  if (!is_root && board.is_repetition()) {
    const int sw = node_count & 1;
    return 2 * sw - 1;
  }
  Value best_score = -EVAL_INFINITE;
  NodeInfo info;
  Move tt_move;
  Move sing_move;
  Move best_move;
  Value tt_value = -EVAL_INFINITE;
  Value sing_value = -EVAL_INFINITE;

  if (ply >= MAX_PLY) {
    return evaluate(td, board.get_position(), ply);
  }

  MoveListe liste;

  get_moves(board.get_position(), liste);
  if (liste.length() == 0) {
    return loss(ply);
  }

  if constexpr (is_root) {
    if (td.num_pv_excluded > 0) {
      MoveListe filtered;
      for (int idx = 0; idx < liste.length(); ++idx) {
        bool excl = false;
        for (int e = 0; e < td.num_pv_excluded; ++e) {
          if (td.pv_excluded_moves[e] == liste[idx]) {
            excl = true;
            break;
          }
        }
        if (!excl) {
          filtered.add_move(liste[idx]);
        }
      }
      liste = filtered;
    }
  }

  if (!is_root) {
    alpha = std::max(loss(ply), alpha);
    beta = std::min(-loss(ply + 1), beta);
    if (alpha >= beta) {
      return alpha;
    }
  }

  if (!excluded.is_empty()) {
    MoveListe filtered;
    for (int idx = 0; idx < liste.length(); ++idx)
      if (liste[idx] != excluded)
        filtered.add_move(liste[idx]);
    liste = filtered;
    if (liste.length() == 0)
      return alpha; // can't happen (we gate on length > 1), but safe
  }

  auto key = board.get_current_key();
  int tab_pieces = 0;
#ifdef _WIN32
  tab_pieces = tablebase.num_pieces;
#endif

  bool found_hash = TT.find_hash(key, info);
  bool is_tt_pv = false;

  Value static_eval = -EVAL_INFINITE;

  if (excluded.is_empty()) {
    is_tt_pv = in_pv || (found_hash && info.ttPv);
    td.ttpv_stack[ply] = is_tt_pv;
  }
  // At root we can still use the tt_move for move_ordering
  if (in_pv && found_hash && info.flag != Flag::None && isEval(info.score)) {
    tt_move = info.tt_move;
    tt_value = value_from_tt(info.score, ply, board.get_position());
  }
  if (excluded.is_empty() && !in_pv && found_hash && info.flag != Flag::None &&
      isEval(info.score)) {
    tt_move = info.tt_move;
    tt_value = value_from_tt(info.score, ply, board.get_position());

    if (info.depth >= depth && info.flag != Flag::None) {
      if ((info.flag == TT_LOWER && tt_value >= beta) ||
          (info.flag == TT_UPPER && tt_value <= alpha) ||
          info.flag == TT_EXACT) {
        return tt_value;
      }
    } else if (TT_SHALLOW_MARGIN && info.depth >= depth - 1 &&
               info.flag == TT_LOWER && !isDecesive(tt_value) &&
               tt_value >= beta + TT_SHALLOW_MARGIN) {
      return tt_value; // one ply too shallow, but far above beta
    }
  }
  // No TT cutoffs at PV nodes (as in Stockfish): an exact-entry cutoff there
  // returns without a PV and truncates the reported line, which happens much
  // more often with helper threads filling the TT.

  if (found_hash && liste.length() > 1 && !tt_move.is_empty() &&
      depth >= SE_MIN_DEPTH && info.flag != Flag::None && !is_sing_search &&
      info.depth >= depth - SE_TT_DEPTH_MARGIN && info.flag != TT_UPPER &&
      isEval(info.score) && !isWinningEval(info.score)) {
    sing_move = tt_move;
    sing_value = tt_value;
  }

  if (!board.get_position().has_jumps(board.get_mover())) {
    if (found_hash && std::abs(info.static_eval) < EVAL_INFINITE) {
      static_eval = value_from_tt(info.static_eval, ply, board.get_position());
    } else {
      static_eval = evaluate(td, board.get_position(), ply);
    }
  }
  // Testing reservoir sampling;
  // #ifdef SAMPLING
  /*if (isEval(static_eval) &&
      !board.get_position().has_jumps(board.get_mover())) {
    reservoir.add_sample(board.get_position(), static_eval);
  }
  */
  // #endif

  const auto static_eval_index = ply + 4;
  td.static_evals[static_eval_index] = static_eval;

  bool improving = false;
  bool opponentWorsening = false;

  if (isEval(static_eval)) {
    if (isEval(td.static_evals[static_eval_index - 2])) {
      improving = static_eval > td.static_evals[static_eval_index - 2];
    } else if (isEval(td.static_evals[static_eval_index - 4])) {
      improving = static_eval > td.static_evals[static_eval_index - 4];
    }
    if (isEval(td.static_evals[static_eval_index - 1])) {
      opponentWorsening = static_eval > -td.static_evals[static_eval_index - 1];
    } else if (isEval(td.static_evals[static_eval_index - 3])) {
      opponentWorsening = static_eval > -td.static_evals[static_eval_index - 3];
    }
  }

#ifdef _WIN32

  TB_RESULT result = TB_RESULT::UNKNOWN;
  if (!is_root && excluded.is_empty() &&
      ((result = tablebase.probe(board.get_position())) !=
       TB_RESULT::UNKNOWN)) {

    auto tb_value = (result == TB_RESULT::WIN)    ? -tbloss(ply)
                    : (result == TB_RESULT::LOSS) ? tbloss(ply)
                                                  : 0;
    if (tb_value == 0) {
      return 0;
    }
    if ((tb_value > 0 && tb_value >= beta) ||
        (tb_value < 0 && tb_value <= alpha)) {

      if (board.get_position().piece_count() <= tab_pieces &&
          std::abs(tb_value) >= MAX_EVAL) {

        auto dtw = tablebase.probe_dtw(board.get_position());

        if (dtw.has_value()) {
          auto actual_dtw = ply + dtw.value();
          return (result == TB_RESULT::WIN) ? -loss(actual_dtw)
                                            : loss(actual_dtw);
        }
        const auto mlh = get_mlh_estimate(td, board.get_position());

        auto tb_value = (result == TB_RESULT::WIN) ? -tbloss(ply) + (300 - mlh)
                        : (result == TB_RESULT::LOSS)
                            ? tbloss(ply) - (300 - mlh)
                            : 0;
        return tb_value;
      }
    }
  }
#endif

  if (!is_tt_pv && depth <= RFP_MAX_DEPTH && static_eval >= beta &&
      tt_move.is_empty() &&
      board.get_position().piece_count() > tab_pieces &&
      !board.get_position().has_jumps() && !isWinningEval(static_eval) &&
      (static_eval - RFP_BASE - (RFP_SLOPE - RFP_IMPROVING * improving) * depth >=
       beta)) {
    return RFP_BLEND ? (RFP_BLEND_W * static_eval + (1024 - RFP_BLEND_W) * beta) /
                           1024
                     : static_eval;
  }

  if (!in_pv && depth <= RAZOR_MAX_DEPTH && isEval(static_eval) &&
      static_eval + RAZOR_BASE + RAZOR_QUAD * depth * depth <= alpha &&
      !board.get_position().has_jumps()) {
    return static_eval;
  }

  // a position already above beta counts as improving
  if (IMPROVING_BETA && isEval(static_eval) && static_eval >= beta)
    improving = true;

  int32_t *out = td.policy_state.out_buffer;
  bool computed = false;

  if constexpr (is_root) {
    if (!td.root_prev_best.is_empty())
      tt_move = td.root_prev_best;
  }

  int start_index = 0;
  if (!tt_move.is_empty()) {
    liste.move_to_front(0, tt_move);
    start_index += (liste[0] == tt_move);
  }

  const Color current_color = board.get_mover();
  auto oracle = [&](Move move) {
    if (move.is_capture()) {
      const uint32_t kings_captured = move.captures & board.get_position().K;
      const uint32_t pawns_captured = move.captures & (~board.get_position().K);
      return (int)(Bits::pop_count(kings_captured) * 14 +
                   Bits::pop_count(pawns_captured) * 10);
    }

    if (!computed) {
      out = policy.get_raw_eval(td.policy_state, board.get_position());
      computed = true;
    }

    if (board.get_position().color == BLACK) {
      move = move.flipped();
    }
    auto encoding = move.get_move_encoding();
    auto score = out[encoding];
    return score;
  };

  std::array<float, 40> move_prob{};
  bool have_prob = false;
  constexpr float PC_MIN_P = 0.01f;
  const Value old_alpha = alpha;
  const Value prob_beta = beta + prob_cut + PROBCUT_IMPROVING * improving;

  const bool tt_blocks =
      found_hash && info.depth >= depth - 4 && info.flag != TT_LOWER &&
      isEval(info.score) &&
      value_from_tt(info.score, ply, board.get_position()) < prob_beta;

  float searched_mass = 0.0f;

  const bool fp_node =
      !is_root && !in_pv && excluded.is_empty() && depth <= FP_MAX_DEPTH &&
      isEval(static_eval) &&
      !board.get_position().has_jumps() && // quiet for both sides
      board.get_position().piece_count() >= FP_MIN_PIECES;

  for (auto i = 0; i < liste.length(); ++i) {
    if (i == start_index) {
      liste.sort(board.get_position(), depth, ply, tt_move, start_index,
                 oracle);

      if (computed && !liste[0].is_capture()) {
        const bool flip = board.get_position().color == BLACK;
        float max_l = -1e9f;
        for (int k = 0; k < liste.length(); ++k) {
          Move m = flip ? liste[k].flipped() : liste[k];
          move_prob[k] = out[m.get_move_encoding()] / 127.0f * POLICY_SCALE;
          max_l = std::max(max_l, move_prob[k]);
        }
        float sum = 0.0f;
        for (int k = 0; k < liste.length(); ++k) {
          move_prob[k] = std::exp(move_prob[k] - max_l);
          sum += move_prob[k];
        }
        for (int k = 0; k < liste.length(); ++k)
          move_prob[k] /= sum;
        have_prob = true;

      }
    }
    const Move move = liste[i];

    const auto kings = board.get_position().K;
    // extension in 1/1024 ply; `extension` keeps only its sign for the
    // pruning/LMR decisions below
    int ext_fp = 0;
    if (liste.length() == 1) {
      ext_fp = SINGLE_EXT_FP;
    } else if (in_pv && move.is_capture()) {
      ext_fp = PV_CAPTURE_EXT_FP;
    } else if (move.is_capture() &&
               board.previous().has_jumps(~board.get_mover())) {
      ext_fp = RECAPTURE_EXT_FP;
    }

    if (!is_root && excluded.is_empty() && !sing_move.is_empty() &&
        move == sing_move) {
      const Value s_beta = sing_value - SE_MARGIN * depth;
      Line s_line;
      const Value s = search<NONPV>(td, cutnode, board, ply, s_line, s_beta - 1,
                                    s_beta, (depth - 1) / 2, move, true);
      // se_tried++;
      if (s < s_beta) { // stage 1: plain extension
        ext_fp = SE_EXT_FP;
        if (!in_pv && s < s_beta - SE_DOUBLE_MARGIN &&
            10 * ply < SE_DOUBLE_PLY_X10 * td.root_depth) {
          ext_fp = SE_DOUBLE_EXT_FP;            // stage 3: double extension
        }
      } else if (s_beta >= beta && isEval(s)) { // stage 2: multi-cut
        return s_beta;
      } else if (tt_value >= beta) { // stage 4: negative extension
        ext_fp = -SE_NEG_EXT_FP;
      } else if (SE_NEG_CUTNODE && cutnode) { // stage 4b: at cut nodes
        ext_fp = -SE_NEG_CUT_EXT_FP;
      }
    }
    const int extension = (ext_fp > 0) - (ext_fp < 0);

    if (!in_pv && have_prob && i >= 2 && extension == 0 && !move.is_capture() &&
        depth < LMP_MASS.size() && searched_mass >= LMP_MASS[depth]) {
      continue; // note: searched_mass is not updated for skipped moves
    }

    Line local_pv;
    Value val = -SCORE_INFINITE;

    Depth reduction = Search::reduce(improving, i, depth, ply, board, move,
                                     in_pv, cutnode, move_prob[i], have_prob);

    // reduction is in 1/1024 ply until here
    if (is_tt_pv && !in_pv) {
      reduction -= LMR_TTPV_FP + (tt_value > alpha ? LMR_TTPV_ALPHA_FP : 0) +
                   (info.depth >= depth ? LMR_TTPV_DEPTH_FP : 0);
    } else if (cutnode && move != tt_move && !tt_move.is_empty()) {
      reduction += LMR_CUTNODE_FP;
    }
    // LMR_CARRY_PCT: this percentage of the fractional part of the reduction is not thrown away but
    // carried into the reduced child's depth
    const int lmr_rem = (LMR_CARRY_PCT && reduction > 0 && extension <= 0)
                            ? (reduction % LMR_ONE) * LMR_CARRY_PCT / 100
                            : 0;
    reduction = reduction > 0 ? reduction / LMR_ONE : 0;

    reduction = (extension > 0 || reduction < 0) ? 0 : reduction;
    board.make_move(move);
    TT.prefetch(board.get_current_key());

    if (!in_pv && !isWinningEval(beta) && depth >= PROBCUT_MIN_DEPTH &&
        board.get_position().piece_count() > tab_pieces && !tt_blocks) {
      Line line;
      const int pc_acc =
          node_acc - PROBCUT_RED_FP - PROBCUT_IMP_RED_FP * improving;
      Depth newDepth = std::max(0, depth + fp_whole(pc_acc));
      td.ext_acc[ply + 1] = pc_acc - fp_whole(pc_acc) * LMR_ONE;
      Value board_val = -qs<NONPV>(td, board, ply + 1, line, -prob_beta,
                                   -prob_beta + 1, 0, Move{}, is_sing_search);

      if (board_val >= prob_beta) {
        Value value = -Search::search<NONPV>(td, !cutnode, board, ply + 1, line,
                                             -prob_beta, -prob_beta + 1,
                                             newDepth, Move{}, is_sing_search);

        if (value >= prob_beta) {
          board.undo_move();
          TT.store_hash(false, value_to_tt(value, ply, board.get_position()),
                        value_to_tt(static_eval, ply, board.get_position()),
                        key, TT_LOWER, newDepth,
                        (!move.is_capture()) ? move : Move{}, is_tt_pv);
          return !isDecesive(value) ? (value - prob_cut) : value;
        }
      }
    }

    const int child_acc = node_acc + ext_fp;
    Depth new_depth = std::max(0, depth - 1 + fp_whole(child_acc));
    td.ext_acc[ply + 1] = child_acc - fp_whole(child_acc) * LMR_ONE;

    if (reduction != 0) {

      td.ext_acc[ply + 1] -= lmr_rem;
      val = -Search::search<NONPV>(
          td, true, board, ply + 1, local_pv, -alpha - 1, -alpha,
          std::max(0, new_depth - reduction), Move{}, is_sing_search);
      td.ext_acc[ply + 1] += lmr_rem;

      if (val > alpha) {
        val = -Search::search<NONPV>(td, !cutnode, board, ply + 1, local_pv,
                                     -alpha - 1, -alpha, new_depth, Move{},
                                     is_sing_search);
      }
    } else if (!in_pv || i != 0) {

      val = -Search::search<NONPV>(td, !cutnode, board, ply + 1, local_pv,
                                   -alpha - 1, -alpha, new_depth, Move{},
                                   is_sing_search);
    }

    if (in_pv && (i == 0 || val > alpha)) {
      val = -Search::search<PV>(td, false, board, ply + 1, local_pv, -beta,
                                -alpha, new_depth, Move{}, is_sing_search);
    }

    board.undo_move();

    if (have_prob) {
      searched_mass += move_prob[i];
    };

    if (val > best_score) {
      best_score = val;

      if (val > alpha) {
        best_move = move;

        if constexpr (is_root) {
          if (val > td.root_partial_score) {
            td.root_partial = move;
            td.root_partial_score = val;
          }
        }

        if (val >= beta) {
          best_move = move;
          best_score = val;
#ifdef SEARCH_STATS
          {
            auto &cs = td.cut_stats;
            cs.cutoffs++;
            cs.cut_first += (i == 0);
            cs.cut_tt += (start_index == 1 && i == 0);
            if (computed && i >= start_index && !move.is_capture()) {
              const int k = start_index; // 0: no TT move, 1: TT move failed
              const int rank = i - start_index;
              cs.pol_cutoffs[k]++;
              cs.pol_rank[k][std::min(rank, 3)]++;
              cs.pol_rank_sum[k] += rank;
            }
          }
#endif
          break;
        }

        pv.concat(move, local_pv);
        alpha = val;
        // remaining moves only need to beat a better alpha: search shallower
        if (!is_root && ALPHA_RAISE_RED_FP && depth > ALPHA_RAISE_MIN_D &&
            depth < ALPHA_RAISE_MAX_D && !isDecesive(val)) {
          node_acc -= ALPHA_RAISE_RED_FP;
          depth += fp_whole(node_acc);
          node_acc -= fp_whole(node_acc) * LMR_ONE;
          // SPSA can combine a large reduction with a small ALPHA_RAISE_MIN_D;
          // depth < 0 would divide by zero in the fail-high blend below
          depth = std::max(depth, 0);
        }
        if constexpr (is_root) {
          if (td.id == 0) {
            glob_current_score = val;
            glob.new_move();
          }
        }
      }
    }
  }
  if (excluded.is_empty() && !is_root && best_score >= beta &&
      isEval(best_score) && !isDecesive(best_score) && !isDecesive(beta)) {
    best_score = (best_score * depth + beta) / (depth + 1);
  }
  if (excluded.is_empty() && !is_root) {
    // a node failing low below a tt-pv parent keeps the pv flag
    if (TTPV_PROPAGATE && best_score <= old_alpha && ply > 0)
      is_tt_pv = is_tt_pv || td.ttpv_stack[ply - 1];
    Value tt_value = value_to_tt(best_score, ply, board.get_position());
    Flag flag;
    if (best_score <= old_alpha) {
      flag = TT_UPPER;
    } else if (best_score >= beta) {
      flag = TT_LOWER;
    } else {
      flag = TT_EXACT;
    }
    Move store_move = (best_move.is_capture()) ? Move{} : best_move;

    TT.store_hash(in_pv, tt_value,
                  value_to_tt(static_eval, ply, board.get_position()), key,
                  flag, depth, store_move, is_tt_pv);
  }

  return best_score;
}

template <NodeType type>
Value qs(ThreadData &td, Board &board, Ply ply, Line &pv, Value alpha,
         Value beta, Depth depth, Move excluded, bool is_sing_search) {
  constexpr bool in_pv = (type != NONPV);
  constexpr NodeType next_type = (type == ROOT) ? PV : type;
  pv.clear();
  const uint64_t node_count = td.inc_nodes();
  check_stop(td, node_count);
  if (board.is_repetition()) {
    const int sw = node_count & 1;
    return 2 * sw - 1;
  }

  if (ply >= MAX_PLY) {
    return evaluate(td, board.get_position(), ply);
  }
  if (ply > td.sel_depth)
    td.sel_depth = ply;

  const auto key = board.get_current_key();

  NodeInfo info;
  bool found_hash = TT.find_hash(key, info);

  bool is_tt_pv = in_pv || (found_hash && info.ttPv);
  if (!in_pv && info.depth >= 0 && found_hash && info.flag != Flag::None &&
      isEval(info.score)) {

    const auto tt_value = value_from_tt(info.score, ply, board.get_position());
    if ((info.flag == TT_LOWER && tt_value >= beta) ||
        (info.flag == TT_UPPER && tt_value <= alpha) || info.flag == TT_EXACT) {
      return tt_value;
    }
  }
  Value static_eval = -SCORE_INFINITE;
  if (found_hash && std::abs(info.static_eval) < EVAL_INFINITE) {
    static_eval = value_from_tt(info.static_eval, ply, board.get_position());
  }

  Value bestValue = -SCORE_INFINITE;
  MoveListe moves;
  get_captures(board.get_position(), moves);

  if (moves.length() == 0) {
    if (board.get_position().is_end()) {
      return loss(ply);
    }

    if (depth == 0 && board.get_position().has_jumps(~board.get_mover())) {
      return Search::search<next_type>(td, false, board, ply, pv, alpha, beta,
                                       1, Move{}, is_sing_search);
    }

    Value net_val;
    if (std::abs(static_eval) < EVAL_INFINITE) {
      net_val = static_eval;
    } else {
      net_val = evaluate(td, board.get_position(), ply);
      TT.store_hash(in_pv, -SCORE_INFINITE,
                    value_to_tt(net_val, ply, board.get_position()), key,
                    TT_LOWER, 0, Move{}, is_tt_pv);
    }

    return net_val;
  }

  moves.sort(board.get_position(), depth, ply, Move{}, 0, [&](Move move) {
    const uint32_t kings_captured = move.captures & board.get_position().K;
    const uint32_t pawns_captured = move.captures & (~board.get_position().K);
    return (int)(Bits::pop_count(kings_captured) * 14 +
                 Bits::pop_count(pawns_captured) * 10);
  });
  Value old_alpha = alpha;
  for (int i = 0; i < moves.length(); ++i) {

    Move move = moves[i];
    Line localPV;
    board.make_move(move);
    TT.prefetch(board.get_current_key());
    Value value;
    value = -Search::qs<next_type>(td, board, ply + 1, localPV, -beta, -alpha,
                                   depth - 1, Move{}, is_sing_search);
    board.undo_move();

    if (value > bestValue) {
      bestValue = value;
      if (value > alpha) {
        pv.concat(move, localPV);
        if (value >= beta)
          break;
        alpha = value;
      }
    }
  }
  // fail-high blend in qsearch
  if (QS_BLEND && bestValue >= beta && !isDecesive(bestValue) &&
      !isDecesive(beta))
    bestValue = (QS_BLEND_W * bestValue + (1024 - QS_BLEND_W) * beta) / 1024;
  if (excluded.is_empty()) {
    Value tt_value = value_to_tt(bestValue, ply, board.get_position());
    Flag flag;
    if (bestValue <= old_alpha) {
      flag = TT_UPPER;
    } else if (bestValue >= beta) {
      flag = TT_LOWER;
    } else {
      flag = TT_EXACT;
    }
    TT.store_hash(in_pv, tt_value,
                  value_to_tt(static_eval, ply, board.get_position()), key,
                  flag, 0, Move{}, is_tt_pv);
  }

  return bestValue;
}
inline Value asp_delta(Value s) {
  const Value a = std::abs(s);
  return asp_wind + (a > ASP_THRESHOLD ? (a - ASP_THRESHOLD) / ASP_DIV : 0);
}
Value search_asp(ThreadData &td, Board &board, Value last_score, Depth depth) {
  constexpr int MAX_RESEARCH = 4;
  if (depth >= 3 && isEval(last_score)) {
    Value delta = asp_delta(last_score);
    Value alpha = std::max(last_score - delta, -EVAL_INFINITE);
    Value beta = std::min(last_score + delta, (Value)EVAL_INFINITE);
    int failed_high = 0;
    for (int tries = 0; tries < MAX_RESEARCH; ++tries) {
      Line line;
      // after a fail high the re-search may be shallower (Stockfish-style)
      const int asp_acc = -ASP_FH_RED_FP * failed_high;
      const Depth asp_depth = std::max(1, depth + fp_whole(asp_acc));
      td.ext_acc[0] = asp_acc - fp_whole(asp_acc) * LMR_ONE;
      const Value score = search<ROOT>(td, false, board, 0, line, alpha, beta,
                                       asp_depth, Move{}, false);
      if (score <= alpha) {
        failed_high = 0;
        beta = (alpha + beta) / 2;
        alpha = std::max(score - delta, -EVAL_INFINITE);
        if (tries == MAX_RESEARCH - 2)
          alpha = -EVAL_INFINITE;
      } else if (score >= beta) {
        if (!td.root_partial.is_empty())
          td.root_prev_best = td.root_partial;
        beta = std::min(score + delta, (Value)EVAL_INFINITE);
        if (tries == MAX_RESEARCH - 2)
          beta = EVAL_INFINITE;
        ++failed_high;
      } else {
        td.main_pv = line;
        return score;
      }
      delta = delta * ASP_GROWTH_X100 / 100;
    }
  }
  Line line;
  td.ext_acc[0] = 0;
  const Value value = search<ROOT>(td, false, board, 0, line, -EVAL_INFINITE,
                                   EVAL_INFINITE, depth, Move{}, false);
  td.main_pv = line;
  return value;
}

} // namespace Search
