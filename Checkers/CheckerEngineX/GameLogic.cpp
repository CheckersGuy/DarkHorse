#include "GameLogic.h"
#include "Bits.h"
#include "MGenerator.h"
#include "Network.h"
#include "Position.h"
#include "types.h"
#include <chrono>
#include <cstdint>
#include <unordered_map>
#include <utility>

std::array<Move, 40> pv_excluded_moves;
int num_pv_excluded = 0;
Line mainPV;
uint64_t endTime = 1000000000;
size_t max_nodes_search = 18446744073709551615ull;
uint64_t nodeCounter = 0u;
int rootDepth = 0;
Value last_eval = -INFINITE;
uint64_t total_counter = 0;
uint64_t diff_counter = 0;

uint64_t counter = 0;
uint64_t both_counter = 0;

uint64_t search_start_time = 0ull;
Value glob_current_score = -INFINITE;
uint64_t last_nodes_per_second = 0ull;

std::array<Value, MAX_PLY + 10> static_evals;
SearchGlobal glob;

Network<2048, 32, 32, 1,128> network;

Network<128, 32, 32, 1,12> mlh_net;

Network<1024, 32, 32, 128,12> policy;

#ifdef _WIN32
void egdb_message_callback(char *msg) {
  if (msg == nullptr)
    return;
#ifdef CHECKERBOARD
  if (glob.reply != nullptr) {
    snprintf(glob.reply, 1024, "%s", msg);
  }
#endif
  std::cout << msg << std::endl;
}
#endif

uint64_t current_nps() {
  const double elapsed_sec =
      std::max(1e-9, (double)(getSystemTime() - search_start_time) / 1e9);
  return static_cast<uint64_t>(nodeCounter / elapsed_sec);
}

void SearchGlobal::new_move() {
#ifdef CHECKERBOARD
  if (reply == nullptr)
    return;
  last_nodes_per_second = current_nps();
  snprintf(reply, 1024, "depth %d/%d score %d nodes %llu KN's %llu PV: %s",
           rootDepth, glob.sel_depth, glob_current_score,
           (unsigned long long)nodeCounter,
           (unsigned long long)(last_nodes_per_second / 1000),
           mainPV.toString(20).c_str());
#endif
}

void SearchGlobal::score_update() {
#ifdef CHECKERBOARD
  if (reply == nullptr)
    return;
  last_nodes_per_second = current_nps();
  snprintf(reply, 1024, "depth %d/%d score %d nodes %llu KN's %llu PV: %s",
           rootDepth, glob.sel_depth, glob_current_score,
           (unsigned long long)nodeCounter,
           (unsigned long long)(last_nodes_per_second / 1000),
           mainPV.toString(20).c_str());
#endif
}

int get_mlh_estimate(Position pos) {

  auto out = mlh_net.evaluate(pos, 0, 0);
  auto scaled = static_cast<float>(out) / 127.0;
  scaled = std::max(0, (int)std::round(scaled * 300));
  return scaled;
}

std::array<float, 40> get_probability_distribution(MoveListe &liste,
                                                   Position pos) {

  std::array<float, 40> logits;
  int32_t *out = &policy.out_layer.buffer[0];
  float sum = 0.0f;
  for (int i = 0; i < liste.length(); ++i) {
    if (liste[i].is_capture())
      continue; // captures already scored separately in oracle
    Move scored = (pos.color == BLACK) ? liste[i].flipped() : liste[i];
    const auto index = scored.get_move_encoding();
    logits[i] = out[index] / 127.0f;
    logits[i] = std::exp(logits[i]);
    sum += logits[i];
  }
  for (int i = 0; i < liste.length(); ++i) {
    logits[i] /= sum;
  }
  return logits;
}

Value blend_mlh(Value eval, Position pos) {
  const Value abs_eval = std::abs(eval);
  const Value ramp_lo = 275, ramp_hi = MAX_EVAL;
  const float t = std::clamp(
      static_cast<float>(abs_eval - ramp_lo) / (ramp_hi - ramp_lo), 0.0f, 1.0f);

  const int mlh = get_mlh_estimate(pos);

  const Value bonus = static_cast<Value>(t * (300 - mlh));

  return eval + ((eval >= 0) ? bonus : -bonus);
}

inline Value value_to_tt(Value v, int ply, Position pos) {

  if (!isEval(v)) {
    return v;
  }

  if (std::abs(v) >= MAX_EVAL) {
    return v >= MAX_EVAL ? v + ply : v <= -MAX_EVAL ? v - ply : v;
  }
  return v >= TB_WIN_MAX_PLY ? v + ply : v <= TB_LOSS_MAX_PLY ? v - ply : v;
}

inline Value value_from_tt(Value v, int ply, Position pos) {
  if (!isEval(v)) {
    return v;
  }

  if (std::abs(v) >= MAX_EVAL) {
    return v >= MAX_EVAL ? v - ply : v <= -MAX_EVAL ? v + ply : v;
  }
  return v >= TB_WIN_MAX_PLY ? v - ply : v <= TB_LOSS_MAX_PLY ? v + ply : v;
}

Value evaluate(Position pos, Ply ply) {
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

    const auto mlh = get_mlh_estimate(pos);
    auto tb_value = (result == TB_RESULT::WIN)    ? -tbloss(ply) + (300 - mlh)
                    : (result == TB_RESULT::LOSS) ? tbloss(ply) - (300 - mlh)
                                                  : 0;
    return tb_value;
  }
#endif

  eval = network.evaluate(pos, ply, 0);
  eval = std::clamp(eval, -MAX_EVAL, MAX_EVAL);
  eval = blend_mlh(eval, pos);

  return eval;
}

std::vector<RootMove> searchValueMultiPV(Board board, int numPV, int depth,
                                         uint32_t time, size_t max_nodes,
                                         bool print, std::ostream &stream) {

#ifdef CHECKERBOARD
  tablebase.reply = glob.reply;
#endif
  max_nodes_search = max_nodes;
  glob.sel_depth = 0u;
  TT.age_counter = (TT.age_counter + 1) & 63ull;
  nodeCounter = 0;
  mainPV.clear();

  MoveListe rootList;
  get_moves(board.get_position(), rootList);

  if (rootList.length() == 1) {
    RootMove root;
    root.score = -last_eval;
    root.move = rootList[0];
    return {root};
  }

  const int actualPV = std::min(numPV, rootList.length());
  if (actualPV <= 0) {
    return {};
  }

  std::vector<RootMove> results(actualPV);

  endTime = getSystemTime() + static_cast<uint64_t>(time) * 1000000;
  search_start_time = getSystemTime();
  size_t total_time = 0;
  num_pv_excluded = 0;

  for (int d = 1; d <= depth; ++d) {
    bool stopped = false;
    num_pv_excluded = 0;

    for (int pv_idx = 0; pv_idx < actualPV; ++pv_idx) {
      network.accumulator.refresh();
      mlh_net.accumulator.refresh();
      policy.accumulator.refresh();

      for (auto k = 0; k < static_evals.size(); ++k) {
        static_evals[k] = -INFINITE;
      }

      auto start_time = std::chrono::high_resolution_clock::now();
      try {
        rootDepth = d;
        Value score =
            Search::search_asp(board, results[pv_idx].previous_score, d);
        results[pv_idx].score = score;
        results[pv_idx].pv = mainPV;
        results[pv_idx].move = (mainPV.length() > 0) ? mainPV[0] : Move{};
      } catch (std::string &msg) {
        stopped = true;
        break;
      }
      auto end_time = std::chrono::high_resolution_clock::now();
      total_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time)
                        .count();

      if (!results[pv_idx].move.is_empty()) {
        pv_excluded_moves[num_pv_excluded++] = results[pv_idx].move;
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

    glob_current_score = results[0].score;
    glob.score_update();

    if (print) {
      double time_seconds = (double)total_time / 1000.0;
      const uint64_t nps = current_nps();
      for (int k = 0; k < actualPV; ++k) {
        stream << "info depth " << d << " multipv " << (k + 1) << " score "
               << results[k].score << " time " << time_seconds << " nodes "
               << nodeCounter << " nps " << (nps / 1000) << "KN's pv "
               << results[k].pv.toString(15) << "\n";
      }
      std::cout << std::endl;
    }
  }

  last_eval = results[0].score;

  num_pv_excluded = 0;
  return results;
}

namespace Search {
Depth reduce(bool improving, int move_index, Depth depth, Ply ply, Board &board,
             Move move, bool in_pv, bool cutnode) {

  if (move_index >= 1 && depth >= 2 && !move.is_capture()) {
    const int d_idx = std::min(depth - 1, LMR_MAX_DEPTH - 1);
    const int m_idx = std::min(move_index, LMR_MAX_MOVE_INDEX - 1);

    auto red = LMR_TABLE_2D[d_idx][m_idx];

    if (in_pv) {
      red = std::max(0, red - 1);
    }
    red += !improving;
    return red;
  }
  return 0;
}

template <NodeType type>
Value search(bool cutnode, Board &board, Ply ply, Line &pv, Value alpha,
             Value beta, Depth depth, Move excluded, bool is_sing_search) {

  constexpr bool is_root = (type == ROOT);
  constexpr bool in_pv = (type == ROOT) || (type == PV);
  constexpr NodeType next_type = (type == ROOT) ? PV : type;
  pv.clear();
  nodeCounter++;

  if ((nodeCounter & 511u) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }
#ifdef CHECKERBOARD
  if ((*glob.playnow) != 0) {
    throw std::string{"Checkerboard requested stop_search"};
  }
#endif
  if (nodeCounter >= max_nodes_search) {
    throw std::string{"Time_out"};
  }

  if (depth <= 0) {
    return Search::qs<next_type>(board, ply, pv, alpha, beta, depth, Move{},
                                 is_sing_search);
  }
  if (!is_root && board.is_repetition()) {
    const int sw = nodeCounter & 1;
    return 0;
  }
  Value best_score = -EVAL_INFINITE;
  NodeInfo info;
  Move tt_move;
  Move sing_move;
  Move best_move;
  Value tt_value = -EVAL_INFINITE;
  Value sing_value = -EVAL_INFINITE;

  if (ply >= MAX_PLY) {
    return evaluate(board.get_position(), ply);
  }

  MoveListe liste;

  get_moves(board.get_position(), liste);
  if (liste.length() == 0) {
    return loss(ply);
  }

  if constexpr (is_root) {
    if (num_pv_excluded > 0) {
      MoveListe filtered;
      for (int idx = 0; idx < liste.length(); ++idx) {
        bool excl = false;
        for (int e = 0; e < num_pv_excluded; ++e) {
          if (pv_excluded_moves[e] == liste[idx]) {
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

  auto key = board.get_current_key();
  int tab_pieces = 0;
#ifdef _WIN32
  tab_pieces = tablebase.num_pieces;
#endif

  Value static_eval = -EVAL_INFINITE;

  bool found_hash = TT.find_hash(key, info);
  bool is_tt_pv = false;

  if (excluded.is_empty()) {
    is_tt_pv = in_pv || (found_hash && info.ttPv);
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
    }
  }

  if (found_hash && liste.length() > 1 && info.flag != Flag::None &&
      !is_sing_search && info.depth >= depth - 4 && info.flag != TT_UPPER &&
      isEval(info.score) && !isWinningEval(info.score)) {
    sing_move = tt_move;
    sing_value = tt_value;
  }
  if (!board.get_position().has_jumps(board.get_mover())) {
    if (found_hash && std::abs(info.static_eval) < EVAL_INFINITE) {
      static_eval = value_from_tt(info.static_eval, ply, board.get_position());
    } else {
      static_eval = evaluate(board.get_position(), ply);
    }
  }

  static_evals[ply] = static_eval;

  bool improving = false;
  bool opponentWorsening = false;

  if (isEval(static_eval)) {
    if (isEval(static_evals[ply - 2])) {
      improving = static_eval > static_evals[ply - 2];
    } else if (isEval(static_evals[ply - 4])) {
      improving = static_eval > static_evals[ply - 4];
    }
    if (isEval(static_evals[ply - 1])) {
      opponentWorsening = static_eval > -static_evals[ply - 1];
    } else if (isEval(static_evals[ply - 3])) {
      opponentWorsening = static_eval > -static_evals[ply - 3];
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
        const auto mlh = get_mlh_estimate(board.get_position());

        auto tb_value = (result == TB_RESULT::WIN) ? -tbloss(ply) + (300 - mlh)
                        : (result == TB_RESULT::LOSS)
                            ? tbloss(ply) - (300 - mlh)
                            : 0;
        return tb_value;
      }
    }
  }
#endif
  if (!is_tt_pv && static_eval >= beta && tt_move.is_empty() &&
      board.get_position().piece_count() > tab_pieces &&
      !board.get_position().has_jumps() && !isWinningEval(static_eval) &&
      (static_eval - 20 - (30 - 7 * improving) * depth >= beta)) {
    return static_eval;
  }

  if (!in_pv && isEval(static_eval) &&
      static_eval + 150 + 10 * depth * depth <= alpha &&
      !board.get_position().has_jumps()) {
    return static_eval;
  }

  if (depth >= 7 && in_pv && tt_move.is_empty() &&
      !board.get_position().has_jumps()) {
    depth--;
  }

  int32_t *out = &policy.out_layer.buffer[0];
  bool computed = false;

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
      out = policy.get_raw_eval(board.get_position());
      computed = true;
    }

    if (board.get_position().color == BLACK) {
      move = move.flipped();
    }
    auto encoding = move.get_move_encoding();
    auto score = out[encoding];
    return score;
  };

  const Value old_alpha = alpha;
  const Value prob_beta = beta + prob_cut + 10 * improving;

  for (auto i = 0; i < liste.length(); ++i) {
    if (i == start_index) {
      liste.sort(board.get_position(), depth, ply, tt_move, start_index,
                 oracle);
    }
    const Move move = liste[i];

    const auto kings = board.get_position().K;
    int extension = 0;
    if (liste.length() == 1) {
      extension = 1;
    } else if (in_pv && move.is_capture()) {
      extension = 1;
    } else if (move.is_capture() &&
               board.previous().has_jumps(~board.get_mover())) {
      extension = 1;
    }
    if (!in_pv && depth < LMP_COUNT.size() && i >= LMP_COUNT[depth] &&
        extension == 0 && !move.is_capture()) {
      continue;
    }

    Line local_pv;
    Value val = -INFINITE;

    Depth reduction =
        Search::reduce(improving, i, depth, ply, board, move, in_pv, cutnode);

    if (is_tt_pv && !in_pv) {
      reduction -= 1 + (tt_value > alpha) + (info.depth >= depth);
    } else if (cutnode && move != tt_move && !tt_move.is_empty()) {
      reduction++;
    }

    reduction = (extension > 0 || reduction < 0) ? 0 : reduction;

    board.make_move(move);

    if (!in_pv && !isWinningEval(beta) && depth >= 1 &&
        board.get_position().piece_count() > tab_pieces) {
      Line line;
      Depth newDepth = std::max(0, depth - 4);
      Value board_val = -qs<NONPV>(board, ply + 1, line, -prob_beta,
                                   -prob_beta + 1, 0, Move{}, is_sing_search);

      if (board_val >= prob_beta) {
        Value value = -Search::search<NONPV>(!cutnode, board, ply + 1, line,
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

    Depth new_depth = std::max(0, depth - 1 + extension);

    if (reduction != 0) {

      val = -Search::search<NONPV>(true, board, ply + 1, local_pv, -alpha - 1,
                                   -alpha, std::max(0, new_depth - reduction),
                                   Move{}, is_sing_search);

      if (val > alpha) {
        val = -Search::search<NONPV>(!cutnode, board, ply + 1, local_pv,
                                     -alpha - 1, -alpha, new_depth, Move{},
                                     is_sing_search);
      }
    } else if (!in_pv || i != 0) {

      val =
          -Search::search<NONPV>(!cutnode, board, ply + 1, local_pv, -alpha - 1,
                                 -alpha, new_depth, Move{}, is_sing_search);
    }

    if (in_pv && (i == 0 || val > alpha)) {
      val = -Search::search<PV>(false, board, ply + 1, local_pv, -beta, -alpha,
                                new_depth, Move{}, is_sing_search);
    }

    board.undo_move();
    if (val > best_score) {
      best_score = val;

      if (val > alpha) {
        best_move = move;
        if (val >= beta) {
          best_move = move;
          best_score = val;
          break;
        }

        pv.concat(move, local_pv);
        alpha = val;
        if constexpr (is_root) {
          glob_current_score = val;
          glob.new_move();
        }
      }
    }
  }

  if (excluded.is_empty() && !is_root) {
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
Value qs(Board &board, Ply ply, Line &pv, Value alpha, Value beta, Depth depth,
         Move excluded, bool is_sing_search) {
  constexpr bool in_pv = (type != NONPV);
  constexpr NodeType next_type = (type == ROOT) ? PV : type;
  pv.clear();
  nodeCounter++;
  if ((nodeCounter & 511u) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }
#ifdef CHECKERBOARD
  if ((*glob.playnow) != 0) {
    throw std::string{"Checkerboard requested stop_search"};
  }
#endif
  if (nodeCounter >= max_nodes_search) {
    throw std::string{"Time_out"};
  }
  if (board.is_repetition()) {
    return 0;
  }

  if (ply >= MAX_PLY) {
    return evaluate(board.get_position(), ply);
  }
  if (ply > glob.sel_depth)
    glob.sel_depth = ply;

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
  Value static_eval = -INFINITE;
  if (found_hash && std::abs(info.static_eval) < EVAL_INFINITE) {
    static_eval = value_from_tt(info.static_eval, ply, board.get_position());
  }

  Value bestValue = -INFINITE;
  MoveListe moves;
  get_captures(board.get_position(), moves);

  if (moves.length() == 0) {
    if (board.get_position().is_end()) {
      return loss(ply);
    }

    if (depth == 0 && board.get_position().has_jumps(~board.get_mover())) {
      return Search::search<next_type>(false, board, ply, pv, alpha, beta, 1,
                                       Move{}, is_sing_search);
    }

    Value net_val;
    if (std::abs(static_eval) < EVAL_INFINITE) {
      net_val = static_eval;
    } else {
      net_val = evaluate(board.get_position(), ply);
      TT.store_hash(in_pv, -INFINITE,
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
    value = -Search::qs<next_type>(board, ply + 1, localPV, -beta, -alpha,
                                   depth - 1, Move{}, is_sing_search);
    board.undo_move();

    if (value > bestValue) {
      bestValue = value;
      if (value > alpha) {
        pv.concat(move, localPV);
      }
      if (value >= beta)
        break;
      alpha = value;
    }
  }
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

Value search_asp(Board &board, Value last_score, Depth depth) {
  Value best_score = -INFINITE;
  const int MAX_RESEARCHES = 4;
  if (depth >= 3 && isEval(last_score)) {
    Value margin = asp_wind;
    Value alpha = last_score - margin;
    Value beta = last_score + margin;
    while (margin < MAX_ASP) {
      Line line;

      auto score = search<ROOT>(false, board, 0, line, alpha, beta, depth,
                                Move{}, false);

      if (score <= alpha) {
        margin += margin / 2;
        alpha = std::max(score - margin, -EVAL_INFINITE);
      } else if (score >= beta) {
        margin += margin / 2;
        beta = std::min(score + margin, int(EVAL_INFINITE));
      } else {
        best_score = score;
        mainPV = line;
        return best_score;
      }
    }
  }
  Line line;
  auto value = search<ROOT>(false, board, 0, line, -EVAL_INFINITE,
                            EVAL_INFINITE, depth, Move{}, false);
  best_score = value;
  mainPV = line;
  return best_score;
}

} // namespace Search
