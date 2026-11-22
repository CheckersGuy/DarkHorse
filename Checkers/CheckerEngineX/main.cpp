#include "Bits.h"
#include "CmdParser.h"
#include "GameLogic.h"
#include "MGenerator.h"
#include "Network.h"
#include "Perft.h"
#include "Thread.h"
#include "Params.h"
#include "Transposition.h"
#ifdef _WIN32
#include "egdb.h"
#endif
#include "incbin.h"
#include "types.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>
INCBIN(mlh_net, "sharing_mlh_25.quant");
INCBIN(network, "buckets1024_107.quant");
INCBIN(policy, "sharing_policy_32.quant");

static Value search_value(const Position &pos, int time_ms, size_t max_nodes,
                          int forced_left = 8) {
  Position tmp = pos;
  MoveListe liste;
  get_moves(tmp, liste);

  if (liste.length() == 0)
    return loss(0);

  if (liste.length() == 1) {
    // follow forced moves instead of searching a one-move root
    if (forced_left == 0)
      return evaluate(pos, 0);
    Position next = pos;
    next.make_move(liste[0]);
    return -search_value(next, time_ms, max_nodes, forced_left - 1);
  }

  Board b(pos);
  auto r =
      searchValueMultiPV(b, 1, MAX_PLY, time_ms, max_nodes, false, std::cout);
  return r.front().score;
}

// Book generation: expands 'depth' plies from the board and prints every
// leaf (FEN, no pending jump) whose score lies in [min, max]; with use_abs
// the window applies to |score|, so positions where either side is better
// qualify (unbalanced books). Leaves are scored with a search of
// 'search_depth' plies and each position is scored at most once.
void recurse(Board &board, std::unordered_set<Position> &hashset, int depth,
             Value min, Value max, int search_depth = 1, bool use_abs = false) {

  if (depth == 0) {
    auto it = hashset.find(board.get_position());
    // if we havent evaluated the position before, evaluate it now
    if (it != hashset.end())
      return;
    hashset.insert(board.get_position());
    if (board.get_position().has_jumps())
      return;

    auto root_moves = searchValueMultiPV(board, 1, search_depth, 1000,
                                         10000000, false, std::cout);
    if (root_moves.empty())
      return;
    Value value = root_moves.front().score;
    if (use_abs)
      value = std::abs(value);

    if (value >= min && value <= max) {
      std::cout << board.get_position().get_fen_string() << std::endl;
    }
    return;
  }

  MoveListe liste;
  get_moves(board.get_position(), liste);

  for (auto i = 0; i < liste.length(); ++i) {
    const Move move = liste[i];
    board.make_move(move);
    recurse(board, hashset, depth - 1, min, max, search_depth, use_abs);
    board.undo_move();
  }

  return;
}

void generate_book(int depth, Position pos, Value min_value, Value max_value,
                   int search_depth = 1, bool use_abs = false) {
  std::unordered_set<Position> hashset;
  Board board(pos);
  recurse(board, hashset, depth, min_value, max_value, search_depth, use_abs);
}

struct SearchThread {

  std::thread local_thread;
  bool is_thinking = false; // if true, there is a search in progress
  Board search_board;
  bool stop_thread = false;

  void init() {

    while (!stop_thread) {
      // should accept some sort of search object
    }
  }
};

// Fixed bench positions (--bench without --position): the node count is a
// signature of the engine version, the NPS is used to calibrate time controls
// across machines (see Training/bench_parallel.sh).
static const char *BENCH_FENS[] = {
    "W:W6,17,18,21,22,26,27,28:B7,9,10,11,15,19",
    "W:W5,10,17,18,22,28,29,31:B1,3,11,12,13,19",
    "W:W20,21,22,23,24,32:B3,4,9,10,13,15",
    "W:W21,23,24,25,26,32:B1,2,4,12,14,15",
    "W:W13,17,21,23,24,27,28,29,31,32:B2,3,4,5,6,7,10,12,14,20",
    "W:W13,17,25,28,29,31,32:B1,4,5,6,12,15,20",
    "W:W14,16,17,18,20,22,25,28,31:B5,6,7,9,11,12,13",
    "W:WK2,13,18,19,21,24,30,32:B3,5,6,8,9,12,23,K27",
    "W:WK2,19,21,23,24,26,31:B3,7,9,11,12,16,K29",
    "W:W21,22,23,24,25,26,28,31:B2,5,7,9,10,11,13,16",
    "W:WK3,K12,13,20,21,24:B1,5,6,K15,26,K27",
    "W:W19,22,23,26,27,28,29,30,32:B2,3,4,8,9,10,11,14,20",
    "W:W13,19,21,22,23,24,27,29,30,32:B1,3,5,6,8,10,12,14,15,20",
    "W:WK2,13,20,21,25,26,30:B1,8,10,14,18",
    "W:W13,15,21,24,28,30:B1,5,8,12,14,23",
    "W:W11,13,22,27,29,30,32:B2,4,6,7,9,19,28",
    "W:W16,20,24,26,29,31:B2,4,6,7,11,21",
    "W:W14,21,22,27,28,32:B7,11,12,16,19,20",
    "W:W17,21,22,24,25,26,27,28,30,31:B2,3,4,5,6,9,11,15,16,20",
    "W:W16,18,20,21,24,28,29,32:B2,5,6,7,8,11,13",
    "W:W14,15,17,24,27,32:B3,5,6,8,12,19",
    "W:W17,18,21,22,24,27,31,32:B2,3,4,5,8,10,12,20",
    "W:W14,18,19,21,23,27,30,31:B2,3,4,5,7,9,12,16,20,K29",
    "W:W18,20,23,24,26,32:B6,9,10,11,16,17",
    "W:W18,19,21,22,23,24,25,26,27,29,32:B2,3,4,5,7,9,10,12,13,14,20",
    "W:W13,19,22,23,27,29,32:B2,4,6,8,10,12,20",
    "W:W13,14,15,21,26,27,28,31:B3,5,6,7,8,16,19,20",
    "W:W9,13,19,21,23,25,27,30,31,32:B2,3,5,6,8,10,12,14,15,20",
    "W:W13,17,18,20,22,26,28,29,30,32:B1,3,4,6,7,9,11,12,16,19",
    "W:W17,18,21,24,27,28,29,31:B2,3,4,6,7,12,13,20",
    "W:W13,17,19,21,22,28,31:B6,10,12,14,15,16,24",
    "W:W12,21,22,26,28,29:B1,2,10,11,13,14",
    "W:W14,17,21,22,24,27,28,32:B2,7,10,11,12,15,20",
    "W:W10,20,21,23,24,29,30,31,32:B1,3,4,6,8,9,11,13",
    "W:W19,20,23,25,27,29,30,31:B2,6,7,8,10,11,12,14",
    "W:W11,18,20,22,23,25,27,32:B2,4,5,7,9,16,19,28",
    "W:W18,20,21,22,23,27,28,29,31,32:B2,3,4,6,8,9,11,12,13,15",
    "W:W17,18,21,28,29,30:B1,4,5,11,12,20",
    "W:W21,22,23,24,25,27,28,29,31:B1,2,5,6,8,12,13,14,15,20",
    "W:W7,11,15,17,21,28:B1,5,9,K18,19,20",
};

int main(int argl, const char **argc) {

  // mlh_net.load_permutation_from_array(gmlh_permData, gmlh_permSize);
  //  policy.load_permutation_from_array(gpolicy_permData, gpolicy_permSize);
  //  network.load_permutation_from_array(gnet_permData, gnet_permSize);

  mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
  network.load_from_array(gnetworkData, gnetworkSize);
  policy.load_from_array(gpolicyData, gpolicySize);

  CmdParser parser;
  parser.parse(argl, argc);
  Board board;

  std::vector<int> value_history;
  int time, depth, hash_size, multipv, maxdb;
  size_t max_nodes = 18446744073709551615ull;
  uint64_t adj_seed = getSystemTime();
  if (parser.has_option("seed")) {
    adj_seed ^= std::stoull(parser.as<std::string>("seed"));
  }
  std::string net_file;

  if (parser.has_option("time")) {
    time = parser.as<int>("time");
  } else {
    time = 100;
  }
  if (parser.has_option("nodes")) {
    max_nodes = parser.as<int>("nodes");
  } else {
    max_nodes = 18446744073709551615ull;
  }

  if (parser.has_option("hash_size")) {
    hash_size = parser.as<int>("hash_size");
  } else {
    hash_size = 128;
  }

  if (parser.has_option("depth")) {
    depth = parser.as<int>("depth");
  } else {
    depth = parser.has_option("bench")
                ? (parser.has_option("position") ? 27 : 16)
                : MAX_PLY;
  }

  // Lazy SMP search threads. Data generation and rescoring always search
  // single-threaded: they already run as many parallel engine processes.
  int num_threads =
      parser.has_option("threads") ? parser.as<int>("threads") : 1;
  if (parser.has_option("generate") || parser.has_option("eval-loop")) {
    num_threads = 1;
  }
  set_num_threads(num_threads);

  // tunable search parameters: --list_params, --<NAME> <value>
  if (parser.has_option("list_params")) {
    print_params(std::cout);
    return 0;
  }
  for (auto &p : tune_params()) {
    if (parser.has_option(p.name))
      set_param(p.name, parser.as<int>(p.name));
  }
  apply_params();

  if (parser.has_option("tm")) {
    TM_ENABLED = parser.as<int>("tm");
  }

  if (parser.has_option("policy_scale")) {
    POLICY_SCALE = std::stof(parser.as<std::string>("policy_scale"));
  }

  if (parser.has_option("multipv")) {
    multipv = parser.as<int>("multipv");
  } else {
    multipv = 1;
  }

  if (parser.has_option("maxdb")) {
    maxdb = parser.as<int>("maxdb");
  } else {
    maxdb = 0;
  }

#ifdef _WIN32
  if (maxdb > 0) {
    tablebase.cache_size = 250;
    tablebase.load_table_base("C:\\kr_english_wld", maxdb);
  }

#endif

  // Opening-book generation, driven by Training/Rust Data::create_book:
  // reads one FEN per line from stdin, prints the qualifying positions
  // --book_plies deeper and then "done"; "terminate" ends the process.
  //   --book_plies N         expansion depth (default 6, 0 = score the input)
  //   --book_search_depth D  search depth used to score a position (default 1)
  //   --book_min A --book_max B  score window (default -55..55)
  //   --book_abs 1           window on |score| (unbalanced books)
  //   --book_hash MB         hash size (default 2)
  if (parser.has_option("book")) {
    const int plies =
        parser.has_option("book_plies") ? parser.as<int>("book_plies") : 6;
    const int sdepth = parser.has_option("book_search_depth")
                           ? parser.as<int>("book_search_depth")
                           : 1;
    const int bmin =
        parser.has_option("book_min") ? parser.as<int>("book_min") : -55;
    const int bmax =
        parser.has_option("book_max") ? parser.as<int>("book_max") : 55;
    const bool babs =
        parser.has_option("book_abs") && parser.as<int>("book_abs") != 0;
    const int bhash =
        parser.has_option("book_hash") ? parser.as<int>("book_hash") : 2;
    TT.resize_in_mb(bhash);
    std::string next_line;
    while (std::getline(std::cin, next_line)) {
      if (!next_line.empty() && next_line.back() == '\r')
        next_line.pop_back();
      if (next_line == "terminate")
        return 0;
      if (next_line.empty())
        continue;
      const auto pos = Position::pos_from_fen(next_line);
      generate_book(plies, pos, bmin, bmax, sdepth, babs);
      // tell the driver to send the next position
      std::cout << "done" << std::endl;
    }
    return 0;
  }

  // Benchmark: all bench positions at a fixed depth (default 16), each from an
  // empty TT. Prints total nodes (version signature) and NPS. Before running
  // SPRT/SPSA on another machine, run Training/bench_parallel.sh there and
  // here and scale the time per move by the NPS ratio.
  if (parser.has_option("bench") && !parser.has_option("position")) {
    // small default hash: clearing a large TT 40 times would dominate the
    // timing under parallel load (memory bandwidth); only searches are timed
    TT.resize_in_mb(parser.has_option("hash_size") ? hash_size : 16);
    uint64_t bench_nodes = 0;
    double ms = 0.0;
    for (const char *fen : BENCH_FENS) {
      TT.clear();
      board = Board{};
      board.get_position() = Position::pos_from_fen(fen);
      const auto t0 = std::chrono::steady_clock::now();
      searchValueMultiPV(board, 1, depth, 1000000000, max_nodes, false,
                         std::cout);
      ms += std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0)
                .count();
      bench_nodes += total_nodes();
    }
    std::cout << "bench nodes " << bench_nodes << " time_ms "
              << (uint64_t)ms << " nps "
              << (uint64_t)(bench_nodes / std::max(ms, 1.0) * 1000.0)
              << std::endl;
    return 0;
  }

  if (parser.has_option("search") || parser.has_option("bench"))

  {
    if (parser.has_option("position")) {
      auto pos_string = parser.as<std::string>("position");
      board.get_position() = Position::pos_from_fen(pos_string);
    } else {
      board.get_position() = Position::get_start_position();
    }
    board.get_position().print_position();

    TT.resize_in_mb(hash_size);
    Move best;
    if (parser.has_option("bench")) {
      searchValueMultiPV(board, multipv, depth, time, max_nodes, false,
                         std::cout);

    } else {
      searchValueMultiPV(board, multipv, depth, time, max_nodes, true,
                         std::cout);
    }

    return 0;
  }

  if (parser.has_option("eval-loop")) {

#ifdef _WIN32

    tablebase.cache_size = 250;
    // tablebase.load_dtw_base("C:\\kr_english_dtw", 10);
    tablebase.load_table_base("C:\\kr_english_wld", 8);

#endif
    const bool children = parser.has_option("children");
    const int child_time =
        parser.has_option("child-time") ? parser.as<int>("child-time") : time;

    TT.resize_in_mb(hash_size);
    std::string current;
    while (std::getline(std::cin, current)) {
      if (current == "terminate") {
        std::exit(-1);
      }

      TT.age_counter++;
      const auto pos = Position::pos_from_fen(current);

      const Value root_value = search_value(pos, time, max_nodes);

      if (!children) {
        std::cout << root_value << std::endl; // old protocol
        continue;
      }

      // Positions where the mover must capture have no quiet-move labels worth
      // training on, so skip the child searches there.
      MoveListe liste;
      Position tmp = pos;
      const bool must_capture = pos.has_jumps(pos.get_color());
      if (!must_capture)
        get_moves(tmp, liste);

      std::ostringstream out;
      out << root_value << " " << liste.length();
      for (int i = 0; i < liste.length(); ++i) {
        Move m = liste[i];
        Position child = pos;
        child.make_move(m);
        const Value q = -search_value(child, child_time, max_nodes);

        // same encoding the policy net uses: white's perspective
        const Move scored = (pos.color == BLACK) ? m.flipped() : m;
        out << " " << scored.get_move_encoding() << " " << m.is_capture() << " "
            << q;
      }
      std::cout << out.str() << std::endl;
    }
    return 0;
  }
  if (parser.has_option("generate")) {

#ifdef _WIN32

    tablebase.cache_size = 1500;
    tablebase.load_dtw_base("D:\\kr_english_dtw", 10);
    tablebase.load_table_base("C:\\kr_english_wld", 10);

#endif

    std::string next_line;
    TT.resize_in_mb(16);
    std::vector<Position> rep_history;
    std::vector<int> rep_values;

    // --- draw adjudication settings (all opt-in) ---
    const int adj_draw_count = parser.has_option("adj_draw_count")
                                   ? parser.as<int>("adj_draw_count")
                                   : 0; // 0 => disabled
    const int adj_draw_score = parser.has_option("adj_draw_score")
                                   ? parser.as<int>("adj_draw_score")
                                   : 0;
    const int adj_draw_min_ply = parser.has_option("adj_draw_min_ply")
                                     ? parser.as<int>("adj_draw_min_ply")
                                     : 0;
    const int adj_draw_max_pieces = parser.has_option("adj_draw_max_pieces")
                                        ? parser.as<int>("adj_draw_max_pieces")
                                        : 24;

    const float adj_draw_prob =
        parser.has_option("adj_draw_prob")
            ? std::stof(parser.as<std::string>("adj_draw_prob"))
            : 1.0f; // default: always active, preserves old behavior

    std::mt19937_64 adj_rng(adj_seed);
    std::uniform_real_distribution<float> adj_prob_dist(0.0f, 1.0f);

    const float multi_pv_prob =
        parser.has_option("multi-pv-prob")
            ? std::stof(parser.as<std::string>("multi-pv-prob"))
            : 0.0f; // 0 => disabled, always plays the best move
    const int multi_pv_eval_diff =
        parser.has_option("multi-pv-eval-diff")
            ? parser.as<int>("multi-pv-eval-diff")
            : 0; // max cp the 2nd-best move is allowed to lose vs. best

    const int multi_pv_min_pieces =
        parser.has_option("multi-pv-min-pieces")
            ? parser.as<int>("multi-pv-min-pieces")
            : 0; // below this piece count, never deviate from best move
                 // (keeps us out of tablebase territory)

    std::mt19937_64 multipv_rng(adj_seed ^ 0x9E3779B97F4A7C15ull);

    auto color_to_result = [](Color color) {
      return ((color == BLACK) ? BLACK_WON : WHITE_WON);
    };
    while (std::getline(std::cin, next_line)) {
      if (next_line == "terminate") {
        std::exit(-1);
      }
      reservoir.clear();
      const auto start_pos = Position::pos_from_fen(next_line);
      rep_history.clear();
      rep_values.clear();
      last_eval = -SCORE_INFINITE;
      board = start_pos;
      Result result = UNKNOWN;
      int draw_streak = 0;
      const bool adj_enabled_this_game = adj_prob_dist(adj_rng) < adj_draw_prob;

      for (auto i = 0; i < 500; ++i) {
        Move best;
        MoveListe liste;
        get_moves(board.get_position(), liste);
        if (liste.length() == 0) {
          result = ((board.get_mover() == BLACK) ? WHITE_WON : BLACK_WON);
          break;
        }

        const bool try_multipv_this_move =
            multi_pv_prob > 0.0f &&
            adj_prob_dist(multipv_rng) < multi_pv_prob &&
            board.get_position().piece_count() > multi_pv_min_pieces;
        const int this_move_pv = try_multipv_this_move ? 2 : 1;

        auto root_moves =
            searchValueMultiPV(board, this_move_pv, depth, this_move_pv * time,
                               max_nodes, false, std::cout);

        auto value = root_moves.front().score;
        best = root_moves.front().move;

        if (try_multipv_this_move && root_moves.size() > 1) {
          const Value best_score = root_moves.front().score;
          std::array<Move, 40> candidates;
          std::array<Value, 40> candidates_scores;
          int num_candidates = 0;
          for (auto &rm : root_moves) {
            if (rm.move.is_empty())
              continue;
            const Value diff = best_score - rm.score;
            if (diff < multi_pv_eval_diff) {
              candidates[num_candidates] = rm.move;
              candidates_scores[num_candidates++] = rm.score;
            }
          }

          if (num_candidates > 1) {
            constexpr double temperature =
                127.0; // tune this; same units as eval (cp)
            std::array<double, 40> weights;
            for (int k = 0; k < num_candidates; ++k) {
              const Value diff = best_score - candidates_scores[k]; // >= 0
              weights[k] = std::exp(-static_cast<double>(diff) / temperature);
            }
            std::discrete_distribution<int> pick_dist(
                weights.begin(), weights.begin() + num_candidates);
            const auto best_index = pick_dist(multipv_rng);
            best = candidates[best_index];
          }
        }

        if (best.is_empty()) {
          result = UNKNOWN;
          break;
        }

        // --- draw adjudication: track consecutive near-zero evals ---
        if (adj_draw_count > 0 && adj_enabled_this_game &&
            i >= adj_draw_min_ply &&
            board.get_position().piece_count() <= adj_draw_max_pieces &&
            std::abs(value) <= adj_draw_score) {
          draw_streak++;
        } else {
          draw_streak = 0;
        }

        if (adj_draw_count > 0 && adj_enabled_this_game &&
            draw_streak >= adj_draw_count) {
          result = DRAW;
          break;
        }

        const Position previous = board.get_position();
        board.play_move(best);
        auto count = std::count(rep_history.begin(), rep_history.end(),
                                (rep_history.empty()) ? Position{} : previous);
        if (count >= 2) {
          result = DRAW;
          break;
        }

        rep_history.emplace_back(previous);
        rep_values.emplace_back(value);
      }

      auto res_to_string = [](Result result, Color color) {
        if ((result == BLACK_WON && color == BLACK) ||
            (result == WHITE_WON && color == WHITE)) {
          return "WON";
        } else if ((result == BLACK_WON && color != BLACK) ||
                   (result == WHITE_WON && color != WHITE)) {
          return "LOSS";
        } else if (result == DRAW) {
          return "DRAW";
        } else {
          return "UNKNOWN";
        }
      };

      // sending all the the results back in reverse order
      std::cout << "BEGIN" << std::endl;
      for (int i = 0; i < reservoir.samples.size(); ++i) {
        std::cout << reservoir.samples[i].pos.WP << "!"
                  << reservoir.samples[i].pos.BP << "!"
                  << reservoir.samples[i].pos.K << "!"
                  << (int)reservoir.samples[i].pos.color << "!"
                  << "UNKNOWN" << "!" << reservoir.samples[i].evaluation
                  << std::endl;
      }
      std::cout << "END" << std::endl;
    }
  }

  std::string current;
  while (std::cin >> current) {
    if (current == "init") {
      TT.age_counter = 0u;
      std::string hash_string;
      std::cin >> hash_string;
      const int hash_size = std::stoi(hash_string);
      TT.resize_in_mb(hash_size);
      std::cout << "init_ready"
                << "\n";
    } else if (current == "new_game") {
      last_eval = -SCORE_INFINITE;
      TT.clear();
      TT.age_counter = 0u;
      std::string position;
      std::cin >> position;
      Position pos = Position::pos_from_fen(position);
      board = Board(pos);
      std::cout << "game_ready"
                << "\n";
    } else if (current == "new_move") {
      // opponent made a move and we need to update the board
      Move move;
      std::vector<uint32_t> squares;
      std::string line;
      std::cin >> line;
      while (!line.empty()) {
        if (line == "end_move")
          break;
        squares.emplace_back(std::stoi(line));
        std::cin >> line;
      }
      move.from = 1u << squares[0];
      move.to = 1u << squares[1];
      for (auto i = 2; i < squares.size(); ++i) {
        move.captures |= 1u << squares[i];
      }

      board.play_move(move);
      std::cout << "update_ready"
                << "\n";
    } else if (current == "search") {
      std::string time_string;
      std::cin >> time_string;
      Move bestMove;
      /*searchValue(board, bestMove, MAX_PLY, std::stoi(time_string), false,
                  std::cout);*/

      auto root_moves =
          searchValueMultiPV(board, 1, MAX_PLY, std::stoi(time_string),
                             max_nodes, false, std::cout);
      bestMove = root_moves[0].move;

      std::cout << "new_move"
                << "\n";
      std::cout << std::to_string(bestMove.get_from_index()) << "\n";
      std::cout << std::to_string(bestMove.get_to_index()) << "\n";
      uint32_t captures = bestMove.captures;
      while (captures) {
        std::cout << std::to_string(Bits::bitscan_foward(captures)) << "\n";
        captures &= captures - 1u;
      }
      std::cout << "end_move"
                << "\n";

      board.play_move(bestMove);
      // adding the move to the repetition history for our side

    } else if (current == "terminate") {
      // terminating the program
      break;
    }
    std::cout.flush();
  }
}
