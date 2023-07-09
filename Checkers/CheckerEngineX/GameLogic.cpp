#include "GameLogic.h"
#include "Move.h"
#include "types.h"

Line mainPV;
uint64_t endTime = 1000000000;
uint64_t nodeCounter = 0u;
Value last_eval = -INFINITE;

SearchGlobal glob;
Network network;

void initialize() { initialize(21231231ull); }

void initialize(uint64_t seed) { Zobrist::init_zobrist_keys(seed); }

////////
Value searchValue(Board board, Move &best, int depth, uint32_t time, bool print,
                  std::ostream &stream) {

  // preallocating memory for the SearchStack
  std::unique_ptr<SearchStack[]> search_stack =
      std::make_unique<SearchStack[]>(MAX_PLY);
  std::memset(search_stack.get(), 0, sizeof(SearchStack) * MAX_PLY);

  Statistics::mPicker.clear_scores();
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

  endTime = getSystemTime() + time;

  Value eval = -INFINITE;
  Local local;

  if (depth == 0) {
    // returning q-search
    return Search::qs(search_stack.get(), false, board, mainPV, -INFINITE,
                      INFINITE, 0, 0, board.pCounter);
  }
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
      best_score = Search::search_asp(search_stack.get(), board, eval, i);
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

Depth reduce(int move_index, Depth depth, Board &board, Move move, bool in_pv) {
  if (move_index >= ((in_pv) ? 3 : 1) && depth >= 2 && !move.is_capture()) {
    auto red = (!in_pv && move_index >= 4) ? 2 : 1;
    return red;
  }
  return 0;
}

template <bool is_root>
Value search(SearchStack *ss, bool in_pv, Board &board, Line &pv, Value alpha,
             Value beta, Ply ply, Depth depth, int last_rev, Move previous,
             Move previous_own) {

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
    return Search::qs(ss, in_pv, board, pv, alpha, beta, ply, depth, last_rev);
  }
  Local local;
  local.move = Move{};
  local.previous = previous;
  // ss->previous = previous;
  local.previous_own = previous_own;

  /*
    if (ply > 2 && local.previous_own == (ss - 1)->previous) {
      std::cout << "Same: " << std::endl;
      std::cout << "StackLocal: " << ss[ply].previous << std::endl;
    } else {
      std::cout << "Ply: " << ply << std::endl;
      std::cout << "NotSame" << std::endl;
    }
  */
  Value best_score = -INFINITE;
  NodeInfo info;
  Move tt_move;
  Move sing_move;
  Value sing_score = -INFINITE;

  if (ply >= MAX_PLY) {
    return board.get_mover() * network.evaluate(board.get_position(), ply);
  }

  MoveListe liste;

  get_moves(board.get_position(), liste);
  if (liste.length() == 0) {
    return loss(ply);
  }

  // mate distance pruning
  alpha = std::max(loss(ply), alpha);
  beta = std::min(-loss(ply + 1), beta);
  if (alpha >= beta) {
    return alpha;
  }

  bool found_hash = TT.find_hash(board.get_current_key(), info);
  // At root we can still use the tt_move for move_ordering
  if ((is_root || in_pv) && found_hash && info.flag != Flag::None) {
    tt_move = info.tt_move;
  }
  if (!in_pv && !is_root && found_hash && info.flag != Flag::None) {
    tt_move = info.tt_move;
    auto tt_score = value_from_tt(info.score, ply);
    if (info.depth >= depth && info.flag != Flag::None) {
      if ((info.flag == TT_LOWER && tt_score >= beta) ||
          (info.flag == TT_UPPER && tt_score <= alpha) ||
          info.flag == TT_EXACT) {
        return tt_score;
      }
    }
    // singular move extensions
    if ((info.flag == TT_LOWER || info.flag == TT_EXACT) &&
        info.depth >= depth - 3 && info.score < MATE_IN_MAX_PLY) {
      // checking for legality of the move stored;
      auto it = (!tt_move.is_empty())
                    ? std::find(liste.begin(), liste.end(), tt_move)
                    : liste.end();
      if (it != liste.end()) {
        sing_score = tt_score;
        sing_move = tt_move;
      }
    }
  }

  bool likelyFailLow =
      (found_hash && in_pv && info.flag == TT_UPPER && info.depth >= depth);

  int start_index = 0;
  if (in_pv && ply < mainPV.length()) {
    const Move mv = mainPV[ply];
    bool sucess = liste.put_front(mv);
    start_index += sucess;
  }
  liste.sort(board.get_position(), depth, ply, local.previous,
             local.previous_own, tt_move, start_index);

  int extension = 0;
  if (liste.length() == 1) {
    extension = 1;
  } else if ((in_pv || local.previous.is_capture()) && liste[0].is_capture()) {
    extension = 1;
  }
  Value old_alpha = alpha;

  for (auto i = 0; i < liste.length(); ++i) {
    // moveLoop starts here
    Move move = liste[i];
    if (move == ss[ply].excluded) {
      continue;
    }
    Line local_pv;
    Depth reduction = Search::reduce(i, depth, board, move, in_pv);

    Value val = -INFINITE;

    /*
    if (depth >= 7 && extension == 0 && in_pv && excluded.is_empty() &&
        move == sing_move && sing_score != -INFINITE) {
      Line line;
      Value threshhold = 40;
      Value newBeta = sing_score - threshhold;
      Value value = Search::search<false>(
          sing_move, in_pv, board, line, (newBeta - 1), newBeta, ply, depth - 4,
          last_rev, local.previous, local.previous_own);
      if (value <= alpha) {
        extension = 1;
      }
    }
    */
    if (move.is_capture() || move.is_pawn_move(board.get_position().K)) {
      last_rev = board.pCounter;
    }
    board.make_move(move);
    // setting the 'previous move in the search stack'

    Depth new_depth = depth - 1 + extension;

    if (!in_pv && depth >= 3 && std::abs(beta) < MATE_IN_MAX_PLY) {
      Line line;
      Value newBeta = beta + prob_cut;
      Depth newDepth = std::max(depth - 4, 1);
      Value board_val = -qs(ss + 1, in_pv, board, line, -newBeta, -newBeta + 1,
                            ply + 1, 0, last_rev);
      if (board_val >= newBeta) {
        Value value = -Search::search<false>(
            ss + 1, false, board, local_pv, -(newBeta + 1), -newBeta, ply + 1,
            newDepth, last_rev, move, local.previous);
        if (value >= newBeta) {
          board.undo_move();
          TT.store_hash(value, board.get_current_key(), TT_LOWER, new_depth + 1,
                        move);
          return value;
        }
      }
    }

    if ((in_pv && i != 0) || reduction != 0) {
      val = -Search::search<false>(
          ss + 1, (i == 0) ? in_pv : false, board, local_pv, -alpha - 1, -alpha,
          ply + 1, new_depth - reduction, last_rev, move, local.previous);
      if (val > alpha && val < beta) {
        val = -Search::search<false>(ss + 1, (i == 0) ? in_pv : false, board,
                                     local_pv, -beta, -alpha, ply + 1,
                                     new_depth, last_rev, move, local.previous);
      }
    } else {
      val = -Search::search<false>(ss + 1, (i == 0) ? in_pv : false, board,
                                   local_pv, -beta, -alpha, ply + 1, new_depth,
                                   last_rev, move, local.previous);
    }
    board.undo_move();
    if (val > best_score) {
      best_score = val;
      local.move = move;
      pv.concat(move, local_pv);
      if (best_score >= beta &&
          !board.get_position().has_jumps(board.get_mover()) &&
          liste.length() > 1) {
        Statistics::mPicker.update_scores(board.get_position(), &liste.liste[0],
                                          local.move, local.previous,
                                          local.previous_own, depth);
        // updating killer moves
        auto &killers = Statistics::mPicker.killer_moves;
        for (auto i = 1; i < MAX_KILLERS; ++i) {
          killers[ply][i] = killers[ply][i - 1];
        }
        killers[ply][0] = local.move;
      }
      if (best_score >= beta) {
        break;
      }
      if (val > alpha) {
        alpha = val;
      }
    }
  }
  // storing killer-moves

  // storing tb-entries
  //
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
    Move store_move = (local.move.is_capture()) ? Move{} : local.move;
    TT.store_hash(tt_value, board.get_current_key(), flag, depth, store_move);
  }
  return best_score;
}

Value qs(SearchStack *ss, bool in_pv, Board &board, Line &pv, Value alpha,
         Value beta, Ply ply, Depth depth, int last_rev) {
  pv.clear();
  nodeCounter++;
  if ((nodeCounter & 1023u) == 0u && getSystemTime() >= endTime) {
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
      return Search::search<false>(ss, in_pv, board, pv, alpha, beta, ply, 1,
                                   last_rev, Move{}, Move{});
    }

    return network.evaluate(board.get_position(), ply);
  }

  if (in_pv && ply < mainPV.length()) {
    moves.put_front(mainPV[ply]);
  }
  for (int i = 0; i < moves.length(); ++i) {
    Move move = moves[i];
    Line localPV;
    board.make_move(move);
    Value value =
        -Search::qs(ss, ((i == 0) ? in_pv : false), board, localPV, -beta,
                    -std::max(alpha, bestValue), ply + 1, depth - 1, last_rev);
    board.undo_move();
    if (value > bestValue) {
      bestValue = value;
      pv.concat(move, localPV);
      if (value >= beta)
        break;
    }
  }

  return bestValue;
}

Value search_asp(SearchStack *ss, Board &board, Value last_score, Depth depth) {
  Value best_score = -INFINITE;
  if (depth >= 3 && isEval(last_score)) {
    Value margin = asp_wind;
    Value alpha_margin = margin;
    Value beta_margin = margin;

    while (std::max(alpha_margin, beta_margin) < MAX_ASP) {
      Line line;
      Value alpha = last_score - alpha_margin;
      Value beta = last_score + beta_margin;
      auto score = search<true>(ss, true, board, line, alpha, beta, 0, depth, 0,
                                Move{}, Move{});
      best_score = score;
      if (score <= alpha) {
        alpha_margin += alpha_margin / 2;
      } else if (score >= beta) {
        beta_margin += beta_margin / 2;
      } else {
        mainPV = line;
        return best_score;
      }
    }
  }
  Line line;
  auto value = search<true>(ss, true, board, line, -INFINITE, INFINITE, 0,
                            depth, 0, Move{}, Move{});
  best_score = value;
  mainPV = line;
  return best_score;
}
} // namespace Search
