#include "GameLogic.h"
#include "Bits.h"
#include "types.h"

Line mainPV;
uint64_t endTime = 1000000000;
uint64_t nodeCounter = 0u;
int rootDepth = 0;
Value last_eval = -INFINITE;

SearchGlobal glob;
Network<4096, 32, 32, 1> network;
Network<512, 32, 32, 1> mlh_net;
Network<2048, 32, 32, 128> policy;

int get_mlh_estimate(Position pos) {
  auto out = mlh_net.evaluate(pos, 0, 0);
  auto scaled = static_cast<float>(out) / 127.0;
  scaled = std::max(0, (int)std::round(scaled * 300));
  return scaled;
}

inline Value value_to_tt(Value v, int ply, Position pos) {

  return v >= MATE_IN_MAX_PLY ? v + ply : v <= MATED_IN_MAX_PLY ? v - ply : v;
}

inline Value value_from_tt(Value v, int ply, Position pos) {

  return v >= MATE_IN_MAX_PLY ? v - ply : v <= MATED_IN_MAX_PLY ? v + ply : v;
}

Value evaluate(Position pos, Ply ply) {

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
    auto tb_value = (result == TB_RESULT::WIN)    ? TB_WIN
                    : (result == TB_RESULT::LOSS) ? TB_LOSS
                                                  : 0;
    eval = tb_value;
  } else {
    eval = network.evaluate(pos, ply, 0);
    eval = std::clamp(eval, -600, 600);
  }
#endif

#ifdef __linux__
  eval = network.evaluate(pos, ply, 0);
  eval = std::clamp(eval, -600, 600);
#endif
  if (Bits::pop_count(pos.BP | pos.WP) <= 10 && std::abs(eval) >= 600) {
    if (eval >= 600) {
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

  const Position start_pos = board.get_position();

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
  size_t total_nodes = 0;
  size_t total_time = 0;
  int i;
  double speed = 0;
  Value best_score = -INFINITE;
  for (i = 1; i <= depth; i += 2) {
    network.accumulator.refresh();
    auto start_time = std::chrono::high_resolution_clock::now();
    std::stringstream ss;
    nodeCounter = 0;
    try {
      rootDepth = i;
      best_score = Search::search_asp(board, eval, i);
    } catch (std::string &msg) {
      break;
    }
    total_nodes += nodeCounter;

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        end_time - start_time)
                        .count();
    if (duration > 0)
      speed = (double)nodeCounter / (double)duration;
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
      ss << "Nodes: " << total_nodes << " | ";
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

  if (move_index >= ((in_pv) ? 3 : 1) && !move.is_capture() &&
      !move.is_promotion(board.get_position().K)) {
    auto red = LMR_TABLE[std::min(depth - 1, 29)];
    if (in_pv) {
      red = PV_LMR_TABLE[std::min(depth - 1, 29)];
    }
    red += (!in_pv && move_index >= 6);
    red += cutnode;
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

  if ((nodeCounter & 2047u) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }
  if (!is_root && board.is_repetition()) {
    return 0;
  }
  if (depth <= 0) {
    return Search::qs<next_type>(board, ply, pv, alpha, beta, depth, Move{},
                                 is_sing_search);
  }

  Value best_score = -INFINITE;
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

  Value static_eval = EVAL_INFINITE;

  bool found_hash = TT.find_hash(key, info);

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
      std::abs(info.score) < MATE_IN_MAX_PLY) {
    sing_move = tt_move;
    sing_value = tt_value;
  }
  if (!board.get_position().has_jumps(board.get_mover())) {
    // only store static evaluation in quiet positions
    if (found_hash && info.flag != Flag::None &&
        info.static_eval != EVAL_INFINITE) {
      static_eval = info.static_eval;
    } else {
      static_eval = evaluate(board.get_position(), ply);
    }
  }

#ifdef _WIN32
  auto result = tablebase.probe(board.get_position());
  if (!is_root && excluded.is_empty() && result != TB_RESULT::UNKNOWN) {
    auto tb_value = (result == TB_RESULT::WIN)    ? TB_WIN
                    : (result == TB_RESULT::LOSS) ? TB_LOSS
                                                  : 0;
    if (tb_value == 0) {
      return 0;
    }
    if ((tb_value > 0 && tb_value >= beta) ||
        (tb_value < 0 && tb_value <= alpha)) {

      if (board.get_position().piece_count() <= 10 &&
          std::abs(tb_value) >= 600) {
        if (tb_value >= 600) {
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

  if (cutnode && tt_move.is_empty()) {
    depth = depth - 2;
  }
  if (depth <= 0) {
    return Search::qs<next_type>(board, ply, pv, alpha, beta, depth, Move{},
                                 is_sing_search);
  }

  auto *out = &policy.output.buffer[0];
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
  liste.sort(board.get_position(), depth, ply, tt_move, start_index, oracle);
  const Value old_alpha = alpha;

  const Value prob_beta = beta + prob_cut;
  for (auto i = 0; i < liste.length(); ++i) {
    const Move move = liste[i];
    if (is_sing_search && move == excluded) {
      continue;
    }
    int extension = 0;
    if (liste.length() == 1) {
      extension = 1;
    } else if (in_pv && move.is_capture()) {
      extension = 1;
    } else if (move.is_capture() &&
               board.previous().has_jumps(~board.get_mover())) {
      extension = 1;
    }

    const auto kings = board.get_position().K;
    Line local_pv;
    Value val = -INFINITE;
    if (!is_root && move == sing_move && depth >= 2 && !is_sing_search &&
        !sing_move.is_empty() && extension == 0) {
      Line local_pv;
      Value sing_beta = sing_value - 25;
      Value sing_depth = std::max(0, depth - 4);

      auto val = Search::search<NONPV>(cutnode, board, ply + 1, local_pv,
                                       sing_beta - 1, sing_beta, sing_depth,
                                       sing_move, true);

      if (val < sing_beta) {
        extension = 1;
      } else if (sing_beta >= beta) {
        return sing_beta;
      }
    }
    Depth reduction =
        Search::reduce(i, depth, ply, board, move, in_pv, cutnode);
    reduction = (extension > 0) ? 0 : reduction;
    board.make_move(move);
    TT.prefetch(board.get_current_key());
    int tab_pieces = 0;
#ifdef _WIN32
    tab_pieces = tablebase.num_pieces;
#endif

    if (!in_pv && std::abs(beta) < TB_WIN && depth >= 1 &&
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
                        static_eval, key, TT_LOWER, newDepth,
                        (!move.is_capture()) ? move : Move{});
          return std::abs(value) < TB_WIN ? (value - prob_cut) : value;
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

    TT.store_hash(in_pv, tt_value, static_eval, key, flag, depth, store_move);
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
  if ((nodeCounter & 2047u) == 0u && getSystemTime() >= endTime) {
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

  MoveListe moves;
  get_captures(board.get_position(), moves);
  Value bestValue = -INFINITE;

  if (moves.is_empty()) {
    if (board.get_position().is_end()) {
      return loss(ply);
    }

    if (depth == 0 && board.get_position().has_jumps(~board.get_mover())) {
      return Search::search<next_type>(false, board, ply, pv, alpha, beta, 1,
                                       Move{}, is_sing_search);
    }

    NodeInfo info;
    const auto key = board.get_current_key();
    bool found_hash = TT.find_hash(key, info);
    Value net_val;

    if (found_hash && info.flag != Flag::None &&
        info.static_eval != EVAL_INFINITE) {
      net_val = info.static_eval;
    } else {
      net_val = evaluate(board.get_position(), ply);

      TT.store_hash(in_pv, EVAL_INFINITE, net_val, key, TT_LOWER, 0, Move{});
    }
    if (info.flag == TT_EXACT && std::abs(info.score) < TB_WIN) {
      return value_from_tt(info.score, ply, board.get_position());
    }
    return net_val;
  }
  moves.sort(board.get_position(), depth, ply, Move{}, 0, [&](Move move) {
    const uint32_t kings_captured = move.captures & board.get_position().K;
    const uint32_t pawns_captured = move.captures & (~board.get_position().K);
    return (int)(Bits::pop_count(kings_captured) * 14 +
                 Bits::pop_count(pawns_captured) * 10);
  });
  for (int i = 0; i < moves.length(); ++i) {

    Move move = moves[i];
    Line localPV;
    board.make_move(move);
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

  return bestValue;
}

Value search_asp(Board &board, Value last_score, Depth depth) {
  Value best_score = -INFINITE;
  if (depth >= 5 && isEval(last_score)) {
    Value margin = asp_wind;
    Value alpha = last_score - margin;
    Value beta = last_score + margin;
    while (margin < MAX_ASP) {
      Line line;

      auto score = search<ROOT>(false, board, 0, line, alpha, beta, depth,
                                Move{}, false);
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
} // namespace Search
