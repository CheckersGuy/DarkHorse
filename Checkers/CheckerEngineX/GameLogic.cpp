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

Value searchValue(Board &board, Move &best, int depth, uint32_t time,
                  bool print, std::ostream &stream) {

  // Statistics::mPicker.clear_scores();

  // setting the color of us

  const Position start_pos = board.get_position();

  board.color_us = ~board.get_mover();
  // debug << board.get_position().get_pos_string() << std::endl;
  // debug << "RepSize : " << board.rep_size << std::endl;
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
    return Search::qs<NONPV>(board, 0, mainPV, -INFINITE, INFINITE, 0,
                             board.pCounter, Move{}, false);
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

  // need to reset the board state;
  board.reset(start_pos);

  return eval;
}

namespace Search {

Depth reduce(int move_index, Depth depth, Ply ply, Board &board, Move move,
             bool in_pv) {

  if (move_index >= ((in_pv) ? 3 : 1) && depth >= 2 && !move.is_capture() &&
      !move.is_promotion(board.get_position().K)) {
    auto red = (!in_pv && move_index >= 4) ? 2 : 1;
    return red;
  }
  return 0;
}

template <NodeType type>
Value search(Board &board, Ply ply, Line &pv, Value alpha, Value beta,
             Depth depth, int last_rev, Move excluded, bool is_sing_search) {

  constexpr bool is_root = (type == ROOT);
  constexpr bool in_pv = (type != NONPV);
  constexpr NodeType next_type = (type == ROOT) ? PV : type;
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
    return Search::qs<next_type>(board, ply, pv, alpha, beta, depth, last_rev,
                                 Move{}, is_sing_search);
  }

  Value best_score = -INFINITE;
  NodeInfo info;
  Move tt_move;
  Move sing_move;
  Move best_move;
  Value tt_value = -INFINITE;
  Value sing_value = -INFINITE;

  if (ply >= MAX_PLY) {
    return board.get_mover() * network.evaluate(board.get_position(), ply,
                                                board.pCounter - last_rev);
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
  auto key = board.get_current_key();
  // this needs to be removed
  // and just do not probe tt at all when excluded is not empty

  bool found_hash = TT.find_hash(key, info);
  // At root we can still use the tt_move for move_ordering
  if (in_pv && found_hash && info.flag != Flag::None) {
    tt_move = info.tt_move;
    tt_value = value_from_tt(info.score, ply);

    if (liste.length() > 1 && !is_sing_search && info.depth >= depth - 4 &&
        info.flag != TT_UPPER && std::abs(info.score) < MATE_IN_MAX_PLY) {
      sing_move = tt_move;
      sing_value = tt_value;
    }
  }
  if (excluded.is_empty() && !in_pv && found_hash && info.flag != Flag::None) {
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

  // singular move extensions goes here e

  // do not do recursive singular searches

  int start_index = 0;
  /*if (in_pv && ply < mainPV.length()) {
    const Move mv = mainPV[ply];
    bool sucess = liste.put_front(mv);
    start_index += sucess;
  }
  */

  int extension = 0;
  if (liste.length() == 1) {
    extension = 1;
  } else if (in_pv && liste[0].is_capture()) {
    extension = 1;
  }

  liste.sort(board.get_position(), depth, ply, tt_move, start_index);

  const Value old_alpha = alpha;

  const Value prob_beta = beta + prob_cut;
  const int parent_rev_move = last_rev;
  // singular extension to be continued
  for (auto i = 0; i < liste.length(); ++i) {
    const Move move = liste[i];
    if (is_sing_search && move == excluded) {
      continue;
    }
    TT.prefetch(board.get_current_key());
    const auto kings = board.get_position().K;
    Line local_pv;
    Depth reduction = Search::reduce(i, depth, ply, board, move, in_pv);

    Value val = -INFINITE;

    if (in_pv && !is_root && move == sing_move && depth >= 4 &&
        !is_sing_search && !sing_move.is_empty() && extension == 0) {
      // std::cout << liste.length() << std::endl;
      //  search every move but the singular move from the tt
      //  if the search fails low, the move is likely the only best move in the
      //  position and we extend the search by 1
      Line local_pv;
      Value sing_beta = sing_value - 60;
      Value sing_depth = depth / 2;
      auto val = Search::search<NONPV>(board, ply + 1, local_pv, sing_beta - 1,
                                       sing_beta, sing_depth, last_rev,
                                       sing_move, true);
      // std::cout << val << std::endl;
      // std::cout << "SingBeta: " << sing_beta << std::endl;
      // std::cout << "----------------------" << std::endl;
      if (val < sing_beta) {
        extension = 1;
      } else if (sing_beta >= beta) {
        return sing_beta;
      }
    }

    board.make_move(move);
    if (move.is_capture() || move.is_pawn_move(kings)) {
      last_rev = board.pCounter;
    } else {
      last_rev = parent_rev_move;
    }
    if (!in_pv && depth > 4 && std::abs(beta) < MATE_IN_MAX_PLY) {
      Line line;
      Depth newDepth = depth - 4;
      Value board_val =
          -qs<NONPV>(board, ply + 1, line, -prob_beta, -prob_beta + 1, 0,
                     last_rev, Move{}, is_sing_search);
      if (board_val >= prob_beta) {
        Value value = -Search::search<NONPV>(
            board, ply + 1, local_pv, -prob_beta, -prob_beta + 1, newDepth,
            last_rev, Move{}, is_sing_search);
        if (value >= prob_beta) {
          board.undo_move();

          TT.store_hash(false, value, key, TT_LOWER, newDepth + 1,
                        (!move.is_capture()) ? move : Move{});
          return value - prob_cut;
        }
      }
    }

    Depth new_depth = depth - 1 + extension;

    if (reduction != 0) {
      val = -Search::search<NONPV>(board, ply + 1, local_pv, -alpha - 1, -alpha,
                                   new_depth - reduction, last_rev, Move{},
                                   is_sing_search);

      if (val > alpha) {
        val =
            -Search::search<NONPV>(board, ply + 1, local_pv, -alpha - 1, -alpha,
                                   new_depth, last_rev, Move{}, is_sing_search);
      }
    } else if (!in_pv || i != 0) {
      val = -Search::search<NONPV>(board, ply + 1, local_pv, -alpha - 1, -alpha,
                                   new_depth, last_rev, Move{}, is_sing_search);
    }

    if (in_pv && (i == 0 || val > alpha)) {
      val = -Search::search<PV>(board, ply + 1, local_pv, -beta, -alpha,
                                new_depth, last_rev, Move{}, is_sing_search);
    }

    const auto copy_root = board.get_position();

    if (is_root) {
      if (std::find(board.rep_history.begin(), board.rep_history.end(),
                    copy_root) != board.rep_history.end()) {
        // debug << "Lowered the score due to repetition" << std::endl;
        val = (val) / 2;
      }
    }

    board.undo_move();
    if (val > best_score) {
      best_score = val;
      if (best_score >= beta && !move.is_capture() && liste.length() > 1) {
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
  if (excluded.is_empty() && !is_root) {
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
    TT.store_hash(in_pv, tt_value, key, flag, depth, store_move);
  }
  return best_score;
}
template <NodeType type>
Value qs(Board &board, Ply ply, Line &pv, Value alpha, Value beta, Depth depth,
         int last_rev, Move excluded, bool is_sing_search) {
  constexpr bool in_pv = (type != NONPV);
  constexpr NodeType next_type = (type == ROOT) ? PV : type;
  pv.clear();
  nodeCounter++;
  if ((nodeCounter & 2047u) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }

  if (ply >= MAX_PLY) {
    return board.get_mover() * network.evaluate(board.get_position(), ply,
                                                board.pCounter - last_rev);
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
      return Search::search<next_type>(board, ply, pv, alpha, beta, 1, last_rev,
                                       Move{}, is_sing_search);
    }
    return network.evaluate(board.get_position(), ply,
                            board.pCounter - last_rev);
  }
  bool sucess = false;
  /*if (in_pv && ply < mainPV.length()) {
    sucess = moves.put_front(mainPV[ply]);
  }
  */
  moves.sort(board.get_position(), depth, ply, Move{}, sucess);
  for (int i = 0; i < moves.length(); ++i) {
    Move move = moves[i];
    Line localPV;
    int last_rev = board.pCounter;
    board.make_move(move);
    Value value;
    value = -Search::qs<next_type>(board, ply + 1, localPV, -beta, -alpha,
                                   depth - 1, last_rev, Move{}, is_sing_search);
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

      auto score =
          search<ROOT>(board, 0, line, alpha, beta, depth, 0, Move{}, false);
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
  auto value = search<ROOT>(board, 0, line, -INFINITE, INFINITE, depth, 0,
                            Move{}, false);
  best_score = value;
  mainPV = line;
  return best_score;
}
} // namespace Search
