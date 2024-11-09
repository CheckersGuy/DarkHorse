#include "GameLogic.h"
#include "Bits.h"
#include "Network.h"
#include "types.h"
#include <cstdint>
#include <unordered_map>
#include <utility>

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

SearchGlobal glob;

Network<4096, 32, 32, 1> network;

Network<128, 32, 32, 1> mlh_net;

Network<512, 32, 32, 128> policy;

int get_mlh_estimate(Position pos) {
  auto out = mlh_net.evaluate(pos, 0, 0);
  auto scaled = static_cast<float>(out) / 127.0;
  scaled = std::max(0, (int)std::round(scaled * 300));
  return scaled;
}

inline Value value_to_tt(Value v, int ply, Position pos) {
  if (std::abs(v) >= 500 && pos.piece_count() <= 10) {
    return v >= 500 ? v + ply : v <= -500 ? v - ply : v;
  }
  return v >= TB_WIN_MAX_PLY ? v + ply : v <= TB_LOSS_MAX_PLY ? v - ply : v;
}

inline Value value_from_tt(Value v, int ply, Position pos) {
  if (std::abs(v) >= 500 && pos.piece_count() <= 10) {
    return v >= 500 ? v - ply : v <= -500 ? v + ply : v;
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
  if (result != TB_RESULT::UNKNOWN) {
    auto tb_value = (result == TB_RESULT::WIN)    ? -tbloss(ply)
                    : (result == TB_RESULT::LOSS) ? tbloss(ply)
                                                  : 0;
    eval = tb_value;
  } else {
    eval = network.evaluate(pos, ply, 0);
    eval = std::clamp(eval, -500, 500);
  }
#endif

#ifdef __linux__

  eval = network.evaluate(pos, ply, 0);

  eval = std::clamp(eval, -500, 500);

#endif
  if (Bits::pop_count(pos.BP | pos.WP) <= 10 && std::abs(eval) >= 500) {
    if (eval >= 500) {
      eval += 300;
      eval -= get_mlh_estimate(pos);
    } else {
      eval -= 300;
      eval += get_mlh_estimate(pos);
    }

    return eval;
  }

  return eval;
}

Value searchValue(Board &board, Move &best, int depth, uint32_t time,
                  bool print, std::ostream &stream) {
  return searchValue(board, best, depth, time, 18446744073709551615ull, print,
                     stream);
}
Value searchValue(Board &board, Move &best, int depth, uint32_t time,
                  size_t max_nodes, bool print, std::ostream &stream) {

  const Position start_pos = board.get_position();
  max_nodes_search = max_nodes;
  glob.sel_depth = 0u;
  TT.age_counter = (TT.age_counter + 1) & 63ull;
  network.accumulator.refresh();
  mlh_net.accumulator.refresh();
  nodeCounter = 0;
  mainPV.clear();
  MoveListe liste;
  get_moves(board.get_position(), liste);
  if (liste.length() == 1) {
    best = liste[0];
    return last_eval;
  }

  Value eval = -INFINITE;
  Local local;

  if (depth == 0) {
    return Search::qs<NONPV>(board, 0, mainPV, -INFINITE, INFINITE, 0, Move{},
                             false);
  }

  endTime = getSystemTime() + time;
  size_t total_time = 0;
  int i;
  double speed = 0;
  Value best_score = -INFINITE;
  nodeCounter = 0;
  for (i = 1; i <= depth; i += 2) {
    network.accumulator.refresh();
    auto start_time = std::chrono::high_resolution_clock::now();
    std::stringstream ss;
    size_t prev_nodes = nodeCounter;
    try {
      rootDepth = i;
      best_score = Search::search_asp(board, eval, i);
    } catch (std::string &msg) {
      break;
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end_time - start_time)
                        .count();
    if (duration > 0)
      speed = (double)(nodeCounter - prev_nodes) / (double)duration;
    total_time += std::chrono::duration_cast<std::chrono::milliseconds>(
                      end_time - start_time)
                      .count();
    eval = best_score;
    last_eval = eval;
    best = mainPV.getFirstMove();

    double time_seconds = (double)total_time / 1000.0;
    if (print) {
      std::string temp = std::to_string(eval) + " ";
      ss << eval << " Depth:" << i << " | " << glob.sel_depth << " | ";
      ss << "Nodes: " << nodeCounter << " | ";
      ss << "Time: " << time_seconds << "\n";
      ss << "Speed: " << (int)(1000000.0 * speed) << " " << mainPV.toString()
         << "\n\n";
      stream << ss.str();
    }
#ifdef CHECKERBOARD
    if (i >= 7) {
      std::stringstream reply_stream;
      reply_stream << "depth " << i << "/" << glob.sel_depth;
      reply_stream << " eval " << eval;
      reply_stream << " time " << time_seconds;
      reply_stream << " speed " << (int)(1000000.0 * speed);
      reply_stream << " pv " << mainPV.toString();
      strcpy(glob.reply, reply_stream.str().c_str());
    }
#endif

    if (isMateVal(best_score)) {
      break;
    }
  }
#ifdef CHECKERBOARD
  double time_seconds = (double)total_time / 1000.0;
  std::stringstream reply_stream;
  reply_stream << "depth " << i << "/" << glob.sel_depth;
  reply_stream << " eval " << eval;
  reply_stream << " time " << time_seconds;
  reply_stream << " speed " << (int)(1000000.0 * speed);
  reply_stream << " pv " << mainPV.toString();
  strcpy(glob.reply, reply_stream.str().c_str());
#endif

  // need to reset the board state;
  board.reset(start_pos);

  return eval;
}

namespace Search {

Depth reduce(int move_index, Depth depth, Ply ply, Board &board, Move move,
             bool in_pv, bool cutnode) {

  if (move_index >= 1 && depth >= 2 && !move.is_capture() &&
      !move.is_promotion(board.get_position().K)) {
    auto red = LMR_TABLE[std::min(depth - 1, 31)];
    if (in_pv) {
      red = std::max(0, red - 1);
    }
    red += (move_index >= 3 + in_pv);
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

  if ((nodeCounter & 1023) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }
  if (nodeCounter >= max_nodes_search) {
    throw std::string{"Time_out"};
  }

  if (depth <= 0) {
    return Search::qs<next_type>(board, ply, pv, alpha, beta, depth, Move{},
                                 is_sing_search);
  }
  if (!is_root && board.is_repetition()) {
    const int sw = nodeCounter & 1;
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
    evaluate(board.get_position(), ply);
  }

  MoveListe liste;

  get_moves(board.get_position(), liste);
  if (liste.length() == 0) {
    return loss(ply);
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

  const auto outer_bound = [&](Value score) {
    return !(std::abs(score) >= TB_WIN_MAX_PLY || (std::abs(score) >= 500)) &&
           (std::abs(score) < EVAL_INFINITE);
  };

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
      std::abs(info.score) < -TB_LOSS_MAX_PLY) {
    sing_move = tt_move;
    sing_value = tt_value;
  }
  if (!board.get_position().has_jumps(board.get_mover())) {
    // only store static evaluation in quiet positions
    if (found_hash && info.flag != Flag::None &&
        std::abs(info.static_eval) < EVAL_INFINITE) {
      static_eval = value_from_tt(info.static_eval, ply, board.get_position());
    } else {
      static_eval = evaluate(board.get_position(), ply);
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
          std::abs(tb_value) >= 500) {
        if (tb_value >= 500) {
          tb_value += 300;
          tb_value -= get_mlh_estimate(board.get_position());
        } else {
          tb_value -= 300;
          tb_value += get_mlh_estimate(board.get_position());
        }
        return tb_value;
      }
    }
  }
#endif

  if (!is_tt_pv && static_eval >= beta && tt_move.is_empty() &&
      board.get_position().piece_count() > tab_pieces &&
      !board.get_position().has_jumps() && outer_bound(static_eval) &&
      // static_eval - 50 - 30 * (depth - 1) >= beta)
      (static_eval - 50 - 30 * (depth - 1) >= beta)) {
    return static_eval;
  }

  int32_t *out;
  std::visit([&](auto &output) { out = &output.buffer[0]; },
             policy.layers.back());
  bool computed = false;

  int start_index = 0;
  if (!tt_move.is_empty()) {
    liste.move_to_front(0, tt_move);
    start_index += (liste[0] == tt_move);
  }

  auto oracle = [&](Move move) {
    if (move.is_capture()) {
      const uint32_t kings_captured = move.captures & board.get_position().K;
      const uint32_t pawns_captured = move.captures & (~board.get_position().K);
      return (int)(Bits::pop_count(kings_captured) * 16 +
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
  const Value prob_beta = beta + prob_cut;
  for (auto i = 0; i < liste.length(); ++i) {
    if (i == start_index) {
      liste.sort(board.get_position(), depth, ply, tt_move, start_index,
                 oracle);
    }

    const Move move = liste[i];

    if (is_sing_search && move == excluded) {
      continue;
    }
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

    Line local_pv;
    Value val = -INFINITE;
    if (!is_root && move == sing_move && depth >= 2 && !is_sing_search &&
        !sing_move.is_empty() && extension == 0) {
      Line local_pv;
      Value sing_beta = sing_value - 25;
      Value sing_depth = std::max(1, depth - 4);

      auto val = Search::search<NONPV>(cutnode, board, ply + 1, local_pv,
                                       sing_beta - 1, sing_beta, sing_depth,
                                       sing_move, true);

      if (val < sing_beta) {
        extension = 1;
      } else if (sing_beta >= beta) {
        return sing_beta;
      } else if (sing_value >= beta) {
        extension = -1;
      }
    }
    Depth reduction =
        Search::reduce(i, depth, ply, board, move, in_pv, cutnode);
    if (is_tt_pv && !in_pv) {
      reduction -= 1 + (tt_value > alpha) + (info.depth >= depth);
    } else if (cutnode && move != tt_move && !tt_move.is_empty()) {
      reduction++;
    }
    reduction = (extension > 0 || reduction < 0) ? 0 : reduction;

    board.make_move(move);
    TT.prefetch(board.get_current_key());

    if (!in_pv && outer_bound(beta) && depth >= 1 &&
        board.get_position().piece_count() > tab_pieces) {
      Line line;
      Depth newDepth = std::max(0, depth - 4);
      Value board_val = -qs<NONPV>(board, ply + 1, line, -prob_beta,
                                   -prob_beta + 1, 0, Move{}, is_sing_search);
      if (newDepth == 0 && board_val >= prob_beta) {
        board.undo_move();
        TT.store_hash(false, value_to_tt(board_val, ply, board.get_position()),
                      value_to_tt(static_eval, ply, board.get_position()), key,
                      TT_LOWER, newDepth, (!move.is_capture()) ? move : Move{},
                      is_tt_pv);
        return outer_bound(board_val) ? (board_val - prob_cut) : board_val;
      }

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
          return outer_bound(value) ? (value - prob_cut) : value;
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

    if (is_root) {
      auto last_position = board.get_position();
      for (auto i = 0; i < board.rep_size; ++i) {
        if (board.rep_history[i] == last_position) {
          val = (val) / 2;
          break;
        }
      }
    }

    board.undo_move();
    if (val > best_score) {
      best_score = val;

      if (val > alpha) {
        best_move = move;
        if (val >= beta) {
          break;
        }

        pv.concat(move, local_pv);
        alpha = val;
      }
    }
  }
  if (!in_pv && best_score >= beta && outer_bound(beta) && outer_bound(beta) &&
      outer_bound(best_score)) {
    best_score = (best_score * depth + beta) / (depth + 1);
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
  if ((nodeCounter & 1023u) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }
  if (nodeCounter >= max_nodes_search) {
    throw std::string{"Time_out"};
  }
  if (board.is_repetition()) {
    const int sw = nodeCounter & 1;
    return 2 * sw - 1;
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
    if ((info.flag == TT_LOWER && info.score >= beta) ||
        (info.flag == TT_UPPER && info.score <= alpha) ||
        info.flag == TT_EXACT) {
      return value_from_tt(info.score, ply, board.get_position());
    }
  }
  Value static_eval = -EVAL_INFINITE;
  if (found_hash && std::abs(info.static_eval) < EVAL_INFINITE) {
    static_eval = value_from_tt(info.static_eval, ply, board.get_position());
  }

  Value bestValue = -INFINITE;

  if (board.is_silent_position(board.get_mover())) {
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
      TT.store_hash(in_pv, -EVAL_INFINITE,
                    value_to_tt(net_val, ply, board.get_position()), key,
                    TT_LOWER, 0, Move{}, is_tt_pv);
    }

    return net_val;
  }
  MoveListe moves;
  get_captures(board.get_position(), moves);
  moves.sort(board.get_position(), depth, ply, Move{}, 0, [&](Move move) {
    const uint32_t kings_captured = move.captures & board.get_position().K;
    const uint32_t pawns_captured = move.captures & (~board.get_position().K);
    return (int)(Bits::pop_count(kings_captured) * 16 +
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
  {
    Value tt_value = value_to_tt(bestValue, ply, board.get_position());
    auto flag = (bestValue >= beta) ? TT_LOWER : TT_UPPER;

    TT.store_hash(in_pv, tt_value,
                  value_to_tt(static_eval, ply, board.get_position()),
                  board.get_current_key(), flag, 0, Move{}, is_tt_pv);
  }

  return bestValue;
}

Value search_asp(Board &board, Value last_score, Depth depth) {
  Value best_score = -INFINITE;
  if (depth >= 3 && isEval(last_score)) {
    Value margin = asp_wind;
    Value alpha = last_score - margin;
    Value beta = last_score + margin;
    while (margin < MAX_ASP) {
      Line line;

      auto score =
          search_root(false, board, 0, line, alpha, beta, depth, Move{}, false);
      if (score <= alpha) {
        beta = (alpha + beta) / 2;
        margin *= 2;
        alpha = std::max(last_score - margin, -EVAL_INFINITE);
      } else if (score >= beta) {
        margin *= 2;
        beta = std::min(last_score + margin, int(EVAL_INFINITE));
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

// need to validate if I get the exact same node-count :)
Value search_root(bool cutnode, Board &board, Ply ply, Line &pv, Value alpha,
                  Value beta, Depth depth, Move excluded, bool is_sing_search) {

  pv.clear();
  nodeCounter++;

  if ((nodeCounter & 1023) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }
  if (nodeCounter >= max_nodes_search) {
    throw std::string{"Time_out"};
  }
  if (board.is_repetition()) {
    const int sw = nodeCounter & 1;
    return 2 * sw - 1;
  }
  if (depth <= 0) {
    return Search::qs<PV>(board, ply, pv, alpha, beta, depth, Move{},
                          is_sing_search);
  }

  Value best_score = -EVAL_INFINITE;
  NodeInfo info;
  Move tt_move;
  Move sing_move;
  Move best_move;
  Value tt_value = -EVAL_INFINITE;
  Value sing_value = -EVAL_INFINITE;

  MoveListe liste;

  get_moves(board.get_position(), liste);
  if (liste.length() == 0) {
    return loss(ply);
  }

  auto key = board.get_current_key();
  int tab_pieces = 0;
#ifdef _WIN32
  tab_pieces = tablebase.num_pieces;
#endif
  const auto outer_bound = [&](Value score) {
    return !(
        std::abs(score) >= TB_WIN_MAX_PLY ||
        (std::abs(score) >= 500 && board.get_position().piece_count() <= 10));
  };

  Value static_eval = -EVAL_INFINITE;

  bool found_hash = TT.find_hash(key, info);
  bool is_tt_pv = true;

  // At root we can still use the tt_move for move_ordering
  if (found_hash && info.flag != Flag::None && isEval(info.score)) {
    tt_move = info.tt_move;
    tt_value = value_from_tt(info.score, ply, board.get_position());
  }

  if (!board.get_position().has_jumps(board.get_mover())) {
    // only store static evaluation in quiet positions
    if (found_hash && info.flag != Flag::None &&
        std::abs(info.static_eval) < EVAL_INFINITE) {
      static_eval = value_from_tt(info.static_eval, ply, board.get_position());
    } else {
      static_eval = evaluate(board.get_position(), ply);
    }
  }

  int32_t *out;
  std::visit([&](auto &output) { out = &output.buffer[0]; },
             policy.layers.back());
  bool computed = false;

  int start_index = 0;
  if (!tt_move.is_empty()) {
    liste.move_to_front(0, tt_move);
    start_index += (liste[0] == tt_move);
  }

  auto oracle = [&](Move move) {
    if (move.is_capture()) {
      const uint32_t kings_captured = move.captures & board.get_position().K;
      const uint32_t pawns_captured = move.captures & (~board.get_position().K);
      return (int)(Bits::pop_count(kings_captured) * 16 +
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
  const Value prob_beta = beta + prob_cut;
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
    } else if (move.is_capture()) {
      extension = 1;
    } else if (move.is_capture() &&
               board.previous().has_jumps(~board.get_mover())) {
      extension = 1;
    }

    Line local_pv;
    Value val = -INFINITE;

    Depth reduction = Search::reduce(i, depth, ply, board, move, true, cutnode);

    reduction = (extension > 0 || reduction < 0) ? 0 : reduction;

    board.make_move(move);
    TT.prefetch(board.get_current_key());

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
    } else if (i != 0) {

      val =
          -Search::search<NONPV>(!cutnode, board, ply + 1, local_pv, -alpha - 1,
                                 -alpha, new_depth, Move{}, is_sing_search);
    }

    if (i == 0 || val > alpha) {
      val = -Search::search<PV>(false, board, ply + 1, local_pv, -beta, -alpha,
                                new_depth, Move{}, is_sing_search);
    }

    auto last_position = board.get_position();
    for (auto i = 0; i < board.rep_size; ++i) {
      if (board.rep_history[i] == last_position) {
        val = (val) / 2;
        break;
      }
    }

    board.undo_move();
    if (val > best_score) {
      best_score = val;

      if (val > alpha) {
        best_move = move;
        if (val >= beta) {
          break;
        }

        pv.concat(move, local_pv);
        alpha = val;
      }
    }
  }

  return best_score;
}

} // namespace Search
