#include "GameLogic.h"
#include "MovePicker.h"
#include "types.h"
#include <optional>

Line mainPV;
uint64_t endTime = 1000000000;
uint64_t nodeCounter = 0u;
Value last_eval = -INFINITE;

SearchGlobal glob;
Network network;

////////
Value searchValue(Board board, Move &best, int depth, uint32_t time, bool print,
                  std::ostream &stream) {

  // Statistics::mPicker.clear_scores();
  Statistics::mPicker.decay_scores();
  glob.sel_depth = 0u;
  TT.age_counter = (TT.age_counter + 1) & 63ull;
  network.accumulator.refresh();
  nodeCounter = 0;
  mainPV.clear();

  MoveListe liste;
  get_moves(board.get_position(), liste);
  /*
  if (liste.length() == 1) {
    best = liste[0];
    return last_eval;
  }
  */

  Value eval = -INFINITE;
  Local local;

  if (depth == 0) {
    // returning q-search
    return Search::qs(false, board, mainPV, -INFINITE, INFINITE, 0, 0,
                      board.pCounter);
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

  return eval;
}

namespace Search {

Depth reduce(int move_index, Depth depth, Ply ply, Board &board, Move move,
             bool in_pv) {

  if (move_index >= ((in_pv) ? 3 : 1) && depth >= 2 && !move.is_capture()) {
    auto red = (!in_pv && move_index >= 4) ? 2 : 1;
    return red;
    ;
  }
  return 0;
}

template <bool is_root>
Value search(bool in_pv, Board &board, Line &pv, Value alpha, Value beta,
             Ply ply, Depth depth, int last_rev) {

  pv.clear();
  nodeCounter++;
  // checking time-used

  if ((nodeCounter & 2047u) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }

  if (!is_root && board.is_repetition(last_rev)) {
    return 0;
  }
  if (depth <= 0) {
    return Search::qs(in_pv, board, pv, alpha, beta, ply, depth, last_rev);
  }

  Value best_score = -INFINITE;
  NodeInfo info;
  Move tt_move;
  Move sing_move;
  Move best_move;
  Value tt_value = -INFINITE;
  Value sing_score = -INFINITE;
  Value max_value = INFINITE;

  if (ply >= MAX_PLY) {
    return board.get_mover() * network.evaluate(board.get_position(), ply);
  }

  if (!is_root) {
    alpha = std::max(loss(ply), alpha);
    beta = std::min(-loss(ply + 1), beta);
    if (alpha >= beta) {
      return alpha;
    }
  }

  MoveListe liste;

  get_moves(board.get_position(), liste);
  if (liste.length() == 0) {
    return loss(ply);
  }

  const auto key = board.get_current_key();
  bool found_hash = TT.find_hash(key, info);
  // At root we can still use the tt_move for move_ordering
  if ((is_root || in_pv) && found_hash && info.flag != Flag::None) {
    tt_move = info.tt_move;
    tt_value = value_from_tt(info.score, ply);
  }
  if (!in_pv && !is_root && found_hash && info.flag != Flag::None) {
    tt_move = info.tt_move;
    tt_value = value_from_tt(info.score, ply);

    if (info.depth >= depth && info.flag != Flag::None) {
      if ((info.flag == TT_LOWER && tt_value >= beta) ||
          (info.flag == TT_UPPER && tt_value <= alpha) ||
          info.flag == TT_EXACT) {
        return tt_value;
      }
    }
  }

  int start_index = 0;
  if (in_pv && ply < mainPV.length()) {
    const Move mv = mainPV[ply];
    bool sucess = liste.put_front(mv);
    start_index += sucess;
  }
  liste.sort(board.get_position(), depth, ply, tt_move, start_index);

  int extension = 0;
  if (liste.length() == 1) {
    extension = 1;
  } else if (in_pv && board.previous().has_jumps() && liste[0].is_capture()) {
    extension = 1;
  }
  const Value old_alpha = alpha;

  const Value prob_beta = beta + prob_cut;

  for (auto i = 0; i < liste.length(); ++i) {
    // moveLoop starts here

    Move move = liste[i];
    TT.prefetch(board.get_current_key());
    const auto kings = board.get_position().K;
    Line local_pv;
    Depth reduction = Search::reduce(i, depth, ply, board, move, in_pv);

    Value val = -INFINITE;

    if (move.is_capture() || move.is_pawn_move(kings)) {
      last_rev = board.pCounter;
    }
    board.make_move(move);

    if (!in_pv && depth >= 3 && std::abs(beta) < MATE_IN_MAX_PLY &&
        board.get_position().piece_count() > 6) {
      Line line;
      Depth newDepth = std::max(depth - 4, 1);
      Value board_val = -qs(in_pv, board, line, -prob_beta, -prob_beta + 1,
                            ply + 1, 0, last_rev);
      if (board_val >= prob_beta) {
        Value value =
            -Search::search<false>(false, board, local_pv, -prob_beta,
                                   -prob_beta + 1, ply + 1, newDepth, last_rev);
        if (value >= prob_beta) {
          board.undo_move();
          if (!board.get_position().has_jumps(board.get_mover()) &&
              liste.length() > 1) {
            Statistics::mPicker.update_scores(
                board.get_position(), &liste.liste[0], move, newDepth + 1);
            // updating killer moves
            auto &killers = Statistics::mPicker.killer_moves;
            for (auto i = 1; i < MAX_KILLERS; ++i) {
              killers[ply][i] = killers[ply][i - 1];
            }
            killers[ply][0] = move;
          }

          TT.store_hash(in_pv, value, key, TT_LOWER, newDepth + 1,
                        (!move.is_capture()) ? move : Move{});
          return value;
        }
      }
    }

    Depth new_depth = depth - 1 + extension;
    if ((in_pv && i != 0) || reduction != 0) {
      val = -Search::search<false>(false, board, local_pv, -alpha - 1, -alpha,
                                   ply + 1, new_depth - reduction, last_rev);
      if (val > alpha) {
        val = -Search::search<false>(in_pv, board, local_pv, -beta, -alpha,
                                     ply + 1, new_depth, last_rev);
      }
    } else {
      val = -Search::search<false>((i == 0) ? in_pv : false, board, local_pv,
                                   -beta, -alpha, ply + 1, new_depth, last_rev);
    }
    board.undo_move();
    if (val > best_score) {
      best_score = val;
      if (best_score >= beta &&
          !board.get_position().has_jumps(board.get_mover()) &&
          liste.length() > 1) {
        Statistics::mPicker.update_scores(board.get_position(), &liste.liste[0],
                                          move, depth);
        // updating killer moves
        auto &killers = Statistics::mPicker.killer_moves;
        for (auto i = 1; i < MAX_KILLERS; ++i) {
          killers[ply][i] = killers[ply][i - 1];
        }
        killers[ply][0] = move;
      }
      if (val > alpha) {
        pv.concat(move, local_pv);
        best_move = move;
        alpha = val;
      }
      if (best_score >= beta) {
        break;
      }
    }
  }
  if (!is_root) {
    Value tt_value = value_to_tt(best_score, ply);
    Flag flag;
    if (best_score <= old_alpha) {
      flag = TT_UPPER;
    } else if (best_score >= beta) {
      flag = TT_LOWER;
    } else {
      flag = TT_EXACT;
    }
    Move store_move = (best_move.is_capture()) ? Move{} : best_move;
    TT.store_hash(in_pv, tt_value, board.get_current_key(), flag, depth,
                  store_move);
  }
  return best_score;
}

Value qs(bool in_pv, Board &board, Line &pv, Value alpha, Value beta, Ply ply,
         Depth depth, int last_rev) {
  pv.clear();
  nodeCounter++;
  if ((nodeCounter & 2047u) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }

  if (ply >= MAX_PLY) {
    return board.get_mover() * network.evaluate(board.get_position(), ply);
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
      return Search::search<false>(in_pv, board, pv, alpha, beta, ply, 1,
                                   last_rev);
    }

    return network.evaluate(board.get_position(), ply);
  }
  bool sucess;
  if (in_pv && ply < mainPV.length()) {
    sucess = moves.put_front(mainPV[ply]);
  }
  moves.sort(board.get_position(), depth, ply, Move{}, sucess);
  for (int i = 0; i < moves.length(); ++i) {
    Move move = moves[i];
    Line localPV;
    board.make_move(move);
    Value value = -Search::qs(((i == 0) ? in_pv : false), board, localPV, -beta,
                              -alpha, ply + 1, depth - 1, board.pCounter - 2);
    board.undo_move();
    if (value > bestValue) {
      bestValue = value;
      if (value > alpha) {
        pv.concat(move, localPV);
      }
      if (value >= beta)
        break;
      else {
        alpha = value;
      }
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

      auto score = search<true>(true, board, line, alpha, beta, 0, depth, 0);
      if (score <= alpha) {
        beta = (alpha + beta) / 2;
        margin *= 2;
        alpha = last_score - margin;
      } else if (score >= beta) {
        margin *= 2;
        beta = last_score + margin;
      } else {
        best_score = score;
        mainPV = line;
        return best_score;
      }
    }
  }
  Line line;
  auto value =
      search<true>(true, board, line, -INFINITE, INFINITE, 0, depth, 0);
  best_score = value;
  mainPV = line;
  return best_score;
}
} // namespace Search
