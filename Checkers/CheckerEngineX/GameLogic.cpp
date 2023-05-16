#include "GameLogic.h"
#include "MovePicker.h"
#include "Network.h"
#include "types.h"

Line mainPV;
uint64_t endTime = 1000000000;
uint64_t nodeCounter = 0u;
Value max_value = INFINITE;

SearchGlobal glob;
Network network, network2;

void initialize() { initialize(32134155995143ull); }

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
  auto test_time = getSystemTime();
  for (int i = 1; i <= depth; i += 2) {
    Statistics::mPicker.clear_move_history();
    auto start_time = getSystemTime();
    std::stringstream ss;
    nodeCounter = 0;
    try {
      Search::search_asp(local, board, eval, i);
    } catch (std::string &msg) {
      break;
    }
    total_nodes += nodeCounter;

    auto time = (getSystemTime() - start_time);
    total_time += time;
    eval = local.best_score;
    best = mainPV.getFirstMove();
    if (print) {
      std::string temp = std::to_string(eval) + " ";
      ss << eval << " Depth:" << i << " | " << glob.sel_depth << " | ";
      ss << "Nodes: " << total_nodes << " | ";
      ss << "Time: " << time << "\n";
      ss << "Speed: " << ((time > 0) ? nodeCounter / time : 0) << " "
         << mainPV.toString() << "\n\n";
      ss << "Time needed: " << time << "\n";
      stream << ss.str();
    }

    if (isMateVal(local.best_score)) {
      break;
    }
  }
  if (print) {
    stream << "TotalNodes: " << total_nodes << "\n";
    stream << "TotalTime: " << getSystemTime() - test_time << "\n";
  }
  return eval;
}

namespace Search {

Depth reduce(Local &local, Board &board, Move move, bool in_pv) {
  Depth red = 0;
  if (!in_pv && local.i >= 2) {
    const auto index = std::min(local.depth, (int)LMR_TABLE.size() - 1);
    red = 1; // previous value
    return LMR_TABLE[index];
  }
  return red;
}

Value search(bool in_pv, Board &board, Line &pv, Value alpha, Value beta,
             Ply ply, Depth depth, int last_rev, Move previous,
             Move previous_own, History history) {

  pv.clear();
  nodeCounter++;
  // checking time-used

  if ((nodeCounter & 2047u) == 0u && getSystemTime() >= endTime) {
    throw std::string{"Time_out"};
  }
  // check again if repetition2 = repetition1
  if (ply > 0 && board.is_repetition(last_rev)) {
    return 0;
  }

  if (depth <= 0) {
    return Search::qs(in_pv, board, pv, alpha, beta, ply, depth, last_rev);
  }

  Local local;
  local.best_score = -INFINITE;
  local.sing_score = -INFINITE;
  local.alpha = alpha;
  local.beta = beta;
  local.ply = ply;
  local.depth = depth;
  local.move = Move{};
  local.previous = previous;
  local.previous_own = previous_own;
  local.skip_move = excluded;
  local.sing_move = Move{};

  // setting move_history

  std::copy(history.begin(), history.end(), local.move_history.begin());

  // checking win condition

  NodeInfo info;
  Move tt_move;

  const uint64_t pos_key = board.get_position().key;

  if (ply >= MAX_PLY) {
    return board.get_mover() * network.evaluate(board.get_position(), ply);
  }
  if (local.alpha >= -loss(ply)) {
    return local.alpha;
  };

  MoveListe liste;

  get_moves(board.get_position(), liste);
  if (liste.length() == 0) {
    return loss(ply);
  }
  bool found_hash = TT.find_hash(pos_key, info);
  if (found_hash && info.flag != Flag::None) {
    tt_move = info.tt_move;
    auto tt_score = valueFromTT(info.score, ply);
    if (info.depth >= depth && info.flag != Flag::None) {
      if ((info.flag == TT_LOWER && tt_score >= local.beta) ||
          (info.flag == TT_UPPER && tt_score <= local.alpha) ||
          info.flag == TT_EXACT) {
        return tt_score;
      }
    }

    if ((info.flag == TT_LOWER && isWin(tt_score) && tt_score >= local.beta) ||
        (info.flag == TT_UPPER && isLoss(tt_score) &&
         tt_score <= local.alpha)) {
      return tt_score;
    }
    if (info.flag == TT_LOWER && info.depth >= depth - 4) {
      // checking if move is in list
      auto found =
          (std::find(liste.begin(), liste.end(), info.tt_move) != liste.end());
      if (found) {
        local.sing_move = tt_move;
        local.sing_score = tt_score;
      }
    }
  }
  int start_index = 0;
  if (in_pv && local.ply < mainPV.length()) {
    const Move mv = mainPV[local.ply];
    bool sucess = liste.put_front(mv);
    start_index += sucess;
  }
  liste.sort(board.get_position(), local, tt_move, start_index);

  // move-loop
  Search::move_loop(in_pv, local, board, pv, liste, last_rev);

  // storing tb-entries
  Value tt_value = toTT(local.best_score, ply);
  Flag flag;
  if (local.best_score > local.alpha) {
    flag = TT_LOWER;
  } else if (local.best_score < local.beta) {
    flag = TT_UPPER;
  } else {
    flag = TT_EXACT;
  }
  Move store_move = (local.move.is_capture()) ? Move{} : local.move;
  TT.store_hash(tt_value, pos_key, flag, depth, store_move);

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
      History history;
      return Search::search(in_pv, board, pv, alpha, beta, ply, 1, last_rev,
                            Move{}, Move{}, history);
    }

    bestValue = network.evaluate(board.get_position(), ply);
    return bestValue;
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

Value searchMove(bool in_pv, Move move, Local &local, Board &board, Line &line,
                 int extension, int last_rev) {

  // singular move extensions
  //
  //

  Depth reduction = Search::reduce(local, board, move, in_pv);

  /*
  if(in_pv
      && local.depth>=8
      && move == local.sing_move
      && local.skip_move.is_empty()
      && extension ==0){
    Value margin = 50;
    Value new_alpha = local.sing_score-margin;
    Line new_pv;
    auto value = search(in_pv, board, new_pv, new_alpha-1, new_alpha, local.ply,
  local.depth-4, last_rev, local.previous, local.previous_own, move);
    if(value<=new_alpha){
      std::cout<<"Extended"<<std::endl;
      extension=1;
    }
    std::cout<<"Value: "<<value << "alpha:" <<new_alpha<<std::endl;

  }
*/

  if (move.is_capture() || move.is_pawn_move(board.get_position().K)) {
    last_rev = board.pCounter;
  }

  Value new_alpha = std::max(local.best_score, local.alpha);
  if (local.best_score > local.alpha) {
    local.move = move;
  }

  Value val = -INFINITE;
  Depth new_depth = local.depth - 1 + extension;

  board.make_move(move);
  // now we can set the move history

  for (auto i = 1; i < HIST_LEN; ++i) {
    local.move_history[i] = local.move_history[i - 1];
  }
  local.move_history[0] = move;

  // Needs an update, do not prune if we are in terriroty of tablebases
  if (!in_pv && local.depth > 1 && std::abs(local.beta) < TB_WIN) {

    Value newBeta = local.beta + prob_cut;
    Depth newDepth = std::max(local.depth - 4, 1);
    Value board_val = -qs(in_pv, board, line, -(newBeta + 1), -newBeta,
                          local.ply + 1, newDepth, last_rev);
    if (board_val >= newBeta) {
      Value value = -Search::search(false, board, line, -(newBeta + 1),
                                    -newBeta, local.ply + 1, newDepth, last_rev,
                                    move, local.previous, local.move_history);
      if (value >= newBeta) {
        val = value;
      }
    }
  }

  if (val == -INFINITE) {
    if ((in_pv && local.i != 0) || reduction != 0) {
      val = -Search::search(false, board, line, -new_alpha - 1, -new_alpha,
                            local.ply + 1, new_depth - reduction, last_rev,
                            move, local.previous, local.move_history);
      if (val > new_alpha) {
        val = -Search::search(in_pv, board, line, -local.beta, -new_alpha,
                              local.ply + 1, new_depth, last_rev, move,
                              local.previous, local.move_history);
      }
    } else {
      val = -Search::search(in_pv, board, line, -local.beta, -new_alpha,
                            local.ply + 1, new_depth, last_rev, move,
                            local.previous, local.move_history);
    }
  }
  board.undo_move();
  return val;
}

void move_loop(bool in_pv, Local &local, Board &board, Line &pv,
               MoveListe &liste, int last_rev) {

  const auto num_moves = liste.length();
  int extension = 0;
  if (liste.length() == 1) {
    extension = 1;
  } else if ((in_pv || local.previous.is_capture()) && liste[0].is_capture()) {
    extension = 1;
  }
  local.i = 0;

  while (local.best_score < local.beta && local.i < num_moves) {
    Move move = liste[local.i];

    if (move != local.skip_move) {
      Line local_pv;
      Value value = searchMove(((local.i == 0) ? in_pv : false), move, local,
                               board, local_pv, extension, last_rev);

      if (value > local.best_score) {
        local.move = move;
        local.best_score = value;
        pv.concat(move, local_pv);
      }
    }
    local.i++;
  }

  if (local.best_score >= local.beta && !board.get_position().has_jumps() &&
      liste.length() > 1 && local.skip_move.is_empty()) {
    Statistics::mPicker.update_scores(board.get_position(), &liste.liste[0],
                                      local.move, local.previous,
                                      local.previous_own, local.depth);
    // updating killer moves
    auto &killers = Statistics::mPicker.killer_moves;
    for (auto i = 1; i < MAX_KILLERS; ++i) {
      killers[local.ply][i] = killers[local.ply][i - 1];
    }
    killers[local.ply][0] = local.move;
  }
}

void search_root(Local &local, Line &line, Board &board, Value alpha,
                 Value beta, Depth depth) {
  std::vector<Move> exluded_moves;
  return search_root(local, line, board, alpha, beta, depth, exluded_moves);
}

void search_root(Local &local, Line &line, Board &board, Value alpha,
                 Value beta, Depth depth, std::vector<Move> &exluded_moves) {
  line.clear();
  local.best_score = -INFINITE;
  local.sing_score = -INFINITE;
  local.alpha = alpha;
  local.beta = beta;
  local.ply = 0;
  local.depth = depth;
  local.move = Move{};
  local.previous = Move{};
  local.previous_own = Move{};
  local.sing_move = Move{};
  local.skip_move = Move{};
  MoveListe liste;
  get_moves(board.get_position(), liste);

  // removing the excluded moves from the list

  for (Move m : exluded_moves) {
    liste.remove(m);
  }

  // why am I not using the hash_move ?
  NodeInfo info;
  bool found_hash = TT.find_hash(board.get_current_key(), info);
  Move tt_move;
  if (found_hash) {
    tt_move = info.tt_move;
  }

  auto sucess = liste.put_front(mainPV[0]);
  int start_index = sucess;

  liste.sort(board.get_position(), local, info.tt_move, start_index);

  move_loop(true, local, board, line, liste, board.last_non_rev);
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
      search_root(local, line, board, alpha, beta, depth);
      Value score = local.best_score;

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
  search_root(local, line, board, -INFINITE, INFINITE, depth);
  mainPV = line;
}
} // namespace Search
