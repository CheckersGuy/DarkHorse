#include "GameLogic.h"
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

  Statistics::mPicker.clear_scores();
  glob.sel_depth = 0u;
  TT.age_counter = (TT.age_counter + 1) & 63ull;
  network.accumulator.refresh();
  nodeCounter = 0;
  mainPV.clear();

  MoveListe liste;
  get_moves(board.get_position(), liste);
  if (liste.length() == 1) {
    best = liste[0];
    return last_eval;
  }

  endTime = getSystemTime() + time;

  Value eval = -INFINITE;
  Local local;

  if (depth == 0) {
    // returning q-search
    return Search::qs(false, board, mainPV, -INFINITE, INFINITE, 0, 0,
                      board.pCounter);
  }
  size_t total_nodes = 0;
  size_t total_time = 0;
  int i;
  double speed = 0;
  for (i = 1; i <= depth; i += 2) {

    network.accumulator.refresh();
    auto start_time = std::chrono::high_resolution_clock::now();
    std::stringstream ss;
    nodeCounter = 0;
    try {
      Search::search_asp(local, board, eval, i);
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
    eval = local.best_score;
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

    if (isMateVal(local.best_score)) {
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

  last_eval = eval;

  return eval;
}

namespace Search {

Depth reduce(int move_index, Depth depth, Board &board, Move move, bool in_pv) {
  if (move_index >= (2 + in_pv) && depth >= 2 && !move.is_capture()) {
    const auto index = std::min(depth, (int)LMR_TABLE.size() - 1);
    return LMR_TABLE[index];
  }
  return 0;
}

template <bool is_root>
Value search(bool in_pv, Board &board, Line &pv, Value alpha, Value beta,
             Ply ply, Depth depth, int last_rev, Move previous,
             Move previous_own) {

  pv.clear();
  nodeCounter++;
  // checking time-used

  if ((nodeCounter & 2047u) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }

  if (!is_root && ply > 0 && board.is_repetition(last_rev)) {
    return 0;
  }

  if (depth <= 0) {
    return Search::qs(in_pv, board, pv, alpha, beta, ply, depth, last_rev);
  }

  Local local;
  local.best_score = -INFINITE;
  local.alpha = alpha;
  local.beta = beta;
  local.move = Move{};
  local.previous = previous;
  local.previous_own = previous_own;
  Value best_score = -INFINITE;
  NodeInfo info;
  Move tt_move;
  Move sing_move;
  Value sing_score;

  if (ply >= MAX_PLY) {
    return board.get_mover() * network.evaluate(board.get_position(), ply);
  }
  if (alpha >= -loss(ply)) {
    return local.alpha;
  };

  MoveListe liste;

  get_moves(board.get_position(), liste);
  if (liste.length() == 0) {
    return loss(ply);
  }
  bool found_hash = TT.find_hash(board.get_current_key(), info);
  if (!is_root && found_hash && info.flag != Flag::None) {
    tt_move = info.tt_move;
    auto tt_score = value_from_tt(info.score, ply);
    if (info.depth >= depth && info.flag != Flag::None) {
      if ((info.flag == TT_LOWER && tt_score >= local.beta) ||
          (info.flag == TT_UPPER && tt_score <= local.alpha) ||
          info.flag == TT_EXACT) {
        return tt_score;
      }
    }

    if ((info.flag == TT_LOWER && isWin(tt_score) && tt_score >= local.beta) ||
        (info.flag == TT_UPPER && isLoss(tt_score) && tt_score <= alpha)) {
      return tt_score;
    }
  }
  int start_index = 0;
  if (in_pv && ply < mainPV.length()) {
    const Move mv = mainPV[ply];
    bool sucess = liste.put_front(mv);
    start_index += sucess;
  }
  liste.sort(board.get_position(), depth, ply, local, tt_move, start_index);

  int extension = 0;
  if (liste.length() == 1) {
    extension = 1;
  } else if ((in_pv || local.previous.is_capture()) && liste[0].is_capture()) {
    extension = 1;
  }

  for (auto i = 0; i < liste.length(); ++i) {
    // moveLoop starts here
    Move move = liste[i];
    Line local_pv;
    Depth reduction = Search::reduce(i, depth, board, move, in_pv);
    if (move.is_capture() || move.is_pawn_move(board.get_position().K)) {
      last_rev = board.pCounter;
    }
    Value new_alpha = std::max(local.best_score, local.alpha);

    Value val = -INFINITE;
    Depth new_depth = depth - 1 + extension;

    board.make_move(move);
    if (!in_pv && depth > 3 && std::abs(local.beta) < TB_WIN) {
      Line line;
      Value newBeta = local.beta + prob_cut;
      Depth newDepth = std::max(depth - 4, 1);
      Value board_val = -qs(in_pv, board, line, -(newBeta + 1), -newBeta,
                            ply + 1, 0, last_rev);
      if (board_val >= newBeta) {
        Value value = -Search::search<false>(
            false, board, local_pv, -(newBeta + 1), -newBeta, ply + 1, newDepth,
            last_rev, move, local.previous);
        if (value >= newBeta) {
          board.undo_move();
          TT.store_hash(value, board.get_current_key(), TT_LOWER, new_depth + 1,
                        move);
          return value;
        }
      }
    }

    if (val == -INFINITE) {
      if ((in_pv && i != 0) || reduction != 0) {
        val = -Search::search<is_root>(
            false, board, local_pv, -new_alpha - 1, -new_alpha, ply + 1,
            new_depth - reduction, last_rev, move, local.previous);
        if (val > new_alpha) {
          val = -Search::search<false>(
              (i == 0) ? in_pv : false, board, local_pv, -local.beta,
              -new_alpha, ply + 1, new_depth, last_rev, move, local.previous);
        }
      } else {
        val = -Search::search<false>((i == 0) ? in_pv : false, board, local_pv,
                                     -local.beta, -new_alpha, ply + 1,
                                     new_depth, last_rev, move, local.previous);
      }
    }
    board.undo_move();
    if (val > local.best_score) {
      local.best_score = val;
      local.move = move;
      pv.concat(move, local_pv);
    }
    if (local.best_score >= beta &&
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
    if (local.best_score >= beta) {
      break;
    }
  }
  // storing killer-moves

  // storing tb-entries
  Value tt_value = value_to_tt(local.best_score, ply);
  Flag flag;
  if (local.best_score > local.alpha) {
    flag = TT_LOWER;
  } else if (local.best_score < local.beta) {
    flag = TT_UPPER;
  } else {
    flag = TT_EXACT;
  }
  Move store_move = (local.move.is_capture()) ? Move{} : local.move;
  TT.store_hash(tt_value, board.get_current_key(), flag, depth, store_move);

  return local.best_score;
}

Value qs(bool in_pv, Board &board, Line &pv, Value alpha, Value beta, Ply ply,
         Depth depth, int last_rev) {
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
      return Search::search<false>(in_pv, board, pv, alpha, beta, ply, 1,
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
        -Search::qs(((i == 0) ? in_pv : false), board, localPV, -beta,
                    -std::max(alpha, bestValue), ply + 1, depth - 1, last_rev);
    board.undo_move();
    if (value > bestValue) {
      bestValue = value;
      if (value >= beta)
        break;

      pv.concat(move, localPV);
    }
  }

  return bestValue;
}

void search_asp(Local &local, Board &board, Value last_score, Depth depth) {
  if (depth >= 3 && isEval(last_score)) {
    Value margin = asp_wind;
    Value alpha_margin = margin;
    Value beta_margin = margin;

    while (std::max(alpha_margin, beta_margin) < MAX_ASP) {
      Line line;
      Value alpha = last_score - alpha_margin;
      Value beta = last_score + beta_margin;
      auto score = search<true>(true, board, line, alpha, beta, 0, depth, 0,
                                Move{}, Move{});
      local.best_score = score;
      if (score <= alpha) {
        alpha_margin *= 2;
      } else if (score >= beta) {
        beta_margin *= 2;
      } else {
        mainPV = line;
        return;
      }
    }
  }
  Line line;
  auto value = search<true>(true, board, line, -INFINITE, INFINITE, 0, depth, 0,
                            Move{}, Move{});
  local.best_score = value;
  mainPV = line;
}
} // namespace Search
