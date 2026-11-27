#include <cstring>
#include <fstream>
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
INCBIN(policy, "policy_500M_40ep.quant");
INCBIN(policy_perm, "policy_500M_40ep.perm"); // neuron permutation (sparsity)

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

static void print_help() {
  std::cout << R"HELP(DarkHorse engine (MainEngine / MainEngine-vnni)

Usage: MainEngine [MODE] [OPTIONS]
Options start with "--"; a value follows after a space (--time 100).
Without a mode the engine reads the match protocol from stdin (see below).

MODES
  --help                    this text
  --search                  search one position and print the iterations
                            (--position FEN, default start position)
  --bench                   benchmark: the bench positions at depth 16, each
                            with an empty TT; prints total nodes (version
                            signature), time and nps. With --position FEN:
                            search that position (default depth 27)
  --eval-loop               rescoring: reads one FEN per line from stdin,
                            prints the root value (see RESCORING)
  --generate                self-play data generation: reads one opening FEN
                            per line, plays a game, prints the sampled
                            positions between BEGIN and END (see GENERATION)
  --book                    opening-book generation: reads FENs from stdin,
                            prints the positions --book_plies deeper whose
                            score lies in the window, then "done"
  --list_params             print the tunable search parameters
                            (name,value,min,max,step) and exit
  --policy_eval FILE        hard-label metrics of the policy net on a POL1
                            file (--policy_eval_n N: first N rows)
  --policy_dump FILE        print the policy net's 128 raw outputs per sample
                            of a POL1 file (training-code check)
  --value_dump FILE         print the value net's eval per sample of a POL1
                            file (--value_dump_n N: first N samples)
  "terminate" on stdin ends every stdin-driven mode.

SEARCH
  --time MS                 time per search in ms (default 100)
  --nodes N                 node limit per search (default unlimited)
  --depth D                 depth limit (default 256; bench 16)
  --hash_size MB            transposition table size (default 128)
  --threads N               Lazy SMP search threads (default 1; --generate
                            and --eval-loop always use 1)
  --multipv N               number of principal variations (default 1)
  --tm 0|1                  time management: skip an iteration that is not
                            predicted to finish in time (default 0)
  --<PARAM> V               set a tunable search parameter, names from
                            --list_params (e.g. --RFP_BASE 20)
  --no_mlh                  no moves-left blend in the evaluation
                            (with --eval-loop: root value only)
  --maxdb N                 tablebases up to N pieces (Windows build only)

NETWORKS (the embedded nets are the default)
  --value_file FILE         load a value net (.quant, Network<2048,32,32,1,128>)
  --value_perm FILE         test: reload the embedded value net with a neuron
                            permutation (result must be bit-exact)
  --policy_file FILE        load a policy net (.quant, Network<1024,32,32,128,128>)
  --policy_perm FILE        neuron permutation for --policy_file
  --policy_scale S          policy logit scale (POLICY_SCALE)
  --log_evals FILE          log every N-th evaluated position to FILE
  --log_rate N              rate for --log_evals (default 1000)

RESCORING (--eval-loop)
  --children                also search every quiet child move; prints
                            "root n enc capture score ..." (policy labels)
  --nodes N                 node budget per search (time is only a safety net)
  --nodes_phase A,B,C,D     node budget by pieces: >=17, 13-16, 9-12, <=8
  --time MS / --child-time MS   time budget per root / child search
                            (used without a node budget)
  --no_mlh                  root value without MLH; prints "no_mlh" first
  --label_diag MS           diagnostics: one "diag" line per position with
                            the root bookkeeping and an MS-ms reference search

GENERATION (--generate; driven by "movegen generate")
  --time MS / --nodes N     budget per move (nodes doubled for 2-PV moves;
                            --nodes without --time: node budget only)
  --depth D                 depth limit
  --seed S                  random seed (mixed with the system time)
  --sample_prob P           keep each new search-tree position with prob. P
                            (default 1)
  --sample_cap N            at most N positions per game (default 2500)
  --sample_bands a,b,c,d    extra keep prob. by |static eval|: <=50, 51-200,
                            201-400, >400 (default 1,1,1,1)
  --multi-pv-prob P         probability to search 2 PVs and maybe play the
                            2nd move (default 0)
  --multi-pv-eval-diff CP   max loss of the 2nd move vs the best (default 0)
  --multi-pv-min-pieces N   never deviate at or below N pieces (default 0)
  --adj_draw_count N        adjudicate a draw after N moves with
                            |eval| <= --adj_draw_score (default 0 = off)
  --adj_draw_score CP       draw window (default 0)
  --adj_draw_min_ply N      earliest ply for adjudication (default 0)
  --adj_draw_max_pieces N   only at or below N pieces (default 24)
  --adj_draw_prob P         share of games with adjudication (default 1)

BOOK (--book)
  --book_plies N            expansion depth (default 6, 0 = score the input)
  --book_search_depth D     search depth for the score (default 1)
  --book_min A --book_max B score window (default -55..55)
  --book_abs 1              window on |score| (unbalanced books)
  --book_hash MB            hash size (default 2)

MATCH PROTOCOL (no mode; used by the Train --match tool)
  stdin commands: init HASH_MB | new_game FEN | new_move SQ SQ [CAPTURES] end_move
                  | search MS | terminate

EXAMPLES
  MainEngine --bench
  MainEngine --search --position B:W13,17,20,23,25,26,27,29,30,31,32:B3,4,5,6,7,8,9,10,11,12,16 --time 1000
  MainEngine --policy_eval val.children.samples --policy_file net.quant
)HELP";
}

int main(int argl, const char **argc) {
#if defined(__AVXVNNI__)
  // MainEngine-vnni on a CPU without AVX-VNNI: report instead of crashing
  if (!__builtin_cpu_supports("avxvnni")) {
    std::cerr << "MainEngine-vnni needs a CPU with AVX-VNNI (Intel 12th gen+ / "
                 "AMD Zen 5); use MainEngine (AVX2)" << std::endl;
    return 1;
  }
#endif

  // mlh_net.load_permutation_from_array(gmlh_permData, gmlh_permSize);
  //  policy.load_permutation_from_array(gpolicy_permData, gpolicy_permSize);
  //  network.load_permutation_from_array(gnet_permData, gnet_permSize);

  mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
  network.load_from_array(gnetworkData, gnetworkSize);
  policy.load_permutation_from_array(gpolicy_permData, gpolicy_permSize);
  policy.load_from_array(gpolicyData, gpolicySize);

  CmdParser parser;
  parser.parse(argl, argc);
  if (parser.has_option("help")) {
    print_help();
    return 0;
  }
  // --no_mlh: no moves-left blend (with --eval-loop only for the root value, see there)
  const bool no_mlh = parser.has_option("no_mlh");
  if (no_mlh)
    set_mlh(false);
  // test option: reload the embedded value net with a neuron permutation
  // (raw size_t[2*L1], see load_permutation); the result must be bit-exact
  if (parser.has_option("value_perm")) {
    network.load_permutation(parser.as<std::string>("value_perm"));
    network.load_from_array(gnetworkData, gnetworkSize);
  }
  // replaces the embedded value net (same architecture: Network<2048, 32, 32, 1, 128>),
  // e.g. for SPRTs of a newly trained net (Training/train_value.py)
  if (parser.has_option("value_file"))
    network.load_bucket(parser.as<std::string>("value_file"));
  // --value_dump FILE [--value_dump_n N]: prints the value net's eval (side to move)
  // of the first N samples of a POL1 file, one per line, to compare with the
  // training code (Experimental.ValueNetwork: 127 * output)
  if (parser.has_option("value_dump")) {
    std::ifstream in(parser.as<std::string>("value_dump"), std::ios::binary);
    uint32_t magic = 0;
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&magic), 4);
    in.read(reinterpret_cast<char *>(&count), 8);
    if (!in || magic != 0x504F4C31u) {
      std::cerr << "not a policy-sample file" << std::endl;
      return 1;
    }
    const uint64_t limit = parser.has_option("value_dump_n")
                               ? std::stoull(parser.as<std::string>("value_dump_n"))
                               : count;
    for (uint64_t i = 0; i < std::min(count, limit); ++i) {
      unsigned char rec[18];
      uint8_t n = 0;
      in.read(reinterpret_cast<char *>(rec), 18);
      in.read(reinterpret_cast<char *>(&n), 1);
      in.ignore(4 * n);
      if (!in)
        break;
      Position pos;
      std::memcpy(&pos.WP, rec, 4);
      std::memcpy(&pos.BP, rec + 4, 4);
      std::memcpy(&pos.K, rec + 8, 4);
      pos.color = static_cast<int8_t>(rec[12]) == -1 ? BLACK : WHITE;
      std::cout << network.get_raw_eval(pos)[0] << "\n";
    }
    return 0;
  }
  // --value_regret FILE [--value_regret_n N]: move choice of the value net (embedded
  // or --value_file) on a POL1 file. Only QUIET children count (after the move the
  // opponent has no capture: the static net is only used on quiet positions, the
  // search resolves captures first). Per position with >= 2 quiet children the net
  // evaluates them and "plays" the best-valued one; regret = best quiet-child score -
  // chosen child score (scores clamped to +-500), value_acc = share with regret <= 10
  // (same as Experimental.ValueNetwork value_acc / value_regret)
  if (parser.has_option("value_regret")) {
    std::ifstream in(parser.as<std::string>("value_regret"), std::ios::binary);
    uint32_t magic = 0;
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&magic), 4);
    in.read(reinterpret_cast<char *>(&count), 8);
    if (!in || magic != 0x504F4C31u) {
      std::cerr << "not a policy-sample file" << std::endl;
      return 1;
    }
    const uint64_t limit = parser.has_option("value_regret_n")
                               ? std::stoull(parser.as<std::string>("value_regret_n"))
                               : count;
    uint64_t rows = 0, hits = 0, unmatched = 0;
    double regret_sum = 0.0;
    // breakdown by the root label |value|: <=50, 51-200, 201-500, >500
    std::array<uint64_t, 4> brows{}, bhits{};
    std::array<double, 4> bregret{};
    for (uint64_t i = 0; i < count && rows < limit; ++i) {
      unsigned char rec[18];
      uint8_t n = 0;
      in.read(reinterpret_cast<char *>(rec), 18);
      in.read(reinterpret_cast<char *>(&n), 1);
      std::vector<unsigned char> ch(4 * n);
      in.read(reinterpret_cast<char *>(ch.data()), 4 * n);
      if (!in)
        break;
      Position pos;
      std::memcpy(&pos.WP, rec, 4);
      std::memcpy(&pos.BP, rec + 4, 4);
      std::memcpy(&pos.K, rec + 8, 4);
      pos.color = static_cast<int8_t>(rec[12]) == -1 ? BLACK : WHITE;
      if (pos.color == BLACK)
        pos = pos.get_color_flip(); // encodings are from white's view
      MoveListe liste;
      get_moves(pos, liste);
      double best = -1e9, vmax = -1e9, chosen = -1e9;
      int quiet = 0;
      std::vector<std::pair<double, double>> vs; // (value, score)
      for (int c = 0; c < n; ++c) {
        const int enc = ch[4 * c];
        const bool capture = ch[4 * c + 1] != 0;
        int16_t score;
        std::memcpy(&score, &ch[4 * c + 2], 2);
        if (capture)
          continue;
        int idx = -1;
        for (int k = 0; k < liste.length(); ++k)
          if (!liste[k].is_capture() && liste[k].get_move_encoding() == enc)
            idx = k;
        if (idx < 0) {
          ++unmatched;
          continue;
        }
        Position child = pos;
        child.make_move(liste[idx]);
        if (child.has_jumps(child.get_color()))
          continue; // not quiet: the opponent can capture
        const double v = -network.get_raw_eval(child)[0];
        const double sc = std::clamp<double>(score, -500, 500);
        vs.emplace_back(v, sc);
        best = std::max(best, sc);
        vmax = std::max(vmax, v);
        ++quiet;
      }
      if (quiet < 2)
        continue;
      // ties (equal values) count as a random choice among them: mean score
      double tsum = 0.0;
      int tcnt = 0;
      for (auto &[v, sc] : vs)
        if (v >= vmax) {
          tsum += sc;
          ++tcnt;
        }
      chosen = tsum / tcnt;
      const double r = best - chosen;
      regret_sum += r;
      hits += r <= 10.0;
      ++rows;
      int16_t root_value;
      std::memcpy(&root_value, rec + 16, 2);
      const int av = std::abs((int)root_value);
      const int band = av <= 50 ? 0 : av <= 200 ? 1 : av <= 500 ? 2 : 3;
      ++brows[band];
      bhits[band] += r <= 10.0;
      bregret[band] += r;
    }
    std::cout << "value_regret rows " << rows << " value_acc " << double(hits) / rows
              << " value_regret " << regret_sum / rows << " unmatched " << unmatched
              << std::endl;
    const char *names[4] = {"|root|<=50", "51-200", "201-500", ">500"};
    for (int b = 0; b < 4; ++b)
      if (brows[b])
        std::cout << "  " << names[b] << ": rows " << brows[b] << " value_acc "
                  << double(bhits[b]) / brows[b] << " value_regret "
                  << bregret[b] / brows[b] << std::endl;
    return 0;
  }
  // replaces the embedded policy net, e.g. for SPRTs of a newly trained net
  // (same architecture: Network<1024, 32, 32, 128, 128>)
  if (parser.has_option("policy_file")) {
    // optional neuron permutation (raw size_t[2*L1], see load_permutation);
    // must be set before the weights are loaded
    if (parser.has_option("policy_perm"))
      policy.load_permutation(parser.as<std::string>("policy_perm"));
    policy.load_bucket(parser.as<std::string>("policy_file"));
  }
  // --policy_dump FILE: prints the policy net's 128 raw outputs (white's move
  // encodings) for every sample of a POL1 file, one line per sample (training-code check)
  if (parser.has_option("policy_dump")) {
    std::ifstream in(parser.as<std::string>("policy_dump"), std::ios::binary);
    uint32_t magic = 0;
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&magic), 4);
    in.read(reinterpret_cast<char *>(&count), 8);
    if (!in || magic != 0x504F4C31u) {
      std::cerr << "not a policy-sample file" << std::endl;
      return 1;
    }
    for (uint64_t i = 0; i < count; ++i) {
      unsigned char rec[18];
      uint8_t n = 0;
      in.read(reinterpret_cast<char *>(rec), 18);
      in.read(reinterpret_cast<char *>(&n), 1);
      in.ignore(4 * n);
      if (!in)
        break;
      Position pos;
      std::memcpy(&pos.WP, rec, 4);
      std::memcpy(&pos.BP, rec + 4, 4);
      std::memcpy(&pos.K, rec + 8, 4);
      pos.color = static_cast<int8_t>(rec[12]) == -1 ? BLACK : WHITE;
      const int32_t *out = policy.get_raw_eval(pos);
      for (int j = 0; j < 128; ++j)
        std::cout << out[j] << (j < 127 ? ' ' : '\n');
    }
    return 0;
  }
  // --policy_eval FILE [--policy_eval_n N]: hard-label metrics of the policy
  // net (embedded or --policy_file) on a POL1 policy-sample file, computed
  // like Experimental.PolicyNetwork: rows with >= 2 moves, label = best child
  // score, logits = raw output / 127 (POLICY_SCALE applied), regret clamped
  // to +-500. Positions in the file are white to move, encodings white's view.
  if (parser.has_option("policy_eval")) {
    if (parser.has_option("policy_scale")) // parsed further below otherwise
      POLICY_SCALE = std::stof(parser.as<std::string>("policy_scale"));
    std::ifstream in(parser.as<std::string>("policy_eval"), std::ios::binary);
    uint32_t magic = 0;
    uint64_t count = 0;
    in.read(reinterpret_cast<char *>(&magic), 4);
    in.read(reinterpret_cast<char *>(&count), 8);
    if (!in || magic != 0x504F4C31u) {
      std::cerr << "not a policy-sample file" << std::endl;
      return 1;
    }
    uint64_t limit = parser.has_option("policy_eval_n")
                         ? std::stoull(parser.as<std::string>("policy_eval_n"))
                         : count;
    uint64_t rows = 0, top1 = 0, top3 = 0, tie10 = 0, blunder = 0;
    double pmax_sum = 0.0, ent_sum = 0.0;
    double ce = 0.0, regret = 0.0;
    for (uint64_t i = 0; i < count && rows < limit; ++i) {
      unsigned char rec[18];
      uint8_t n = 0;
      in.read(reinterpret_cast<char *>(rec), 18);
      in.read(reinterpret_cast<char *>(&n), 1);
      struct Ch { uint8_t enc, cap; int16_t score; } ch[64];
      for (int c = 0; c < n; ++c)
        in.read(reinterpret_cast<char *>(&ch[c]), 4);
      if (!in)
        break;
      if (n < 2)
        continue;
      Position pos;
      std::memcpy(&pos.WP, rec, 4);
      std::memcpy(&pos.BP, rec + 4, 4);
      std::memcpy(&pos.K, rec + 8, 4);
      pos.color = static_cast<int8_t>(rec[12]) == -1 ? BLACK : WHITE;
      const int32_t *out = policy.get_raw_eval(pos);
      int best = 0, pred = 0;
      double maxl = -1e30;
      for (int c = 0; c < n; ++c) {
        if (ch[c].score > ch[best].score) best = c;
        const double l = out[ch[c].enc] / 127.0 * POLICY_SCALE;
        if (l > maxl) { maxl = l; pred = c; }
      }
      double z = 0.0;
      for (int c = 0; c < n; ++c)
        z += std::exp(out[ch[c].enc] / 127.0 * POLICY_SCALE - maxl);
      ce += -(out[ch[best].enc] / 127.0 * POLICY_SCALE - maxl - std::log(z));
      pmax_sum += 1.0 / z; // probability of the top-ranked move
      for (int c = 0; c < n; ++c) {
        const double pc =
            std::exp(out[ch[c].enc] / 127.0 * POLICY_SCALE - maxl) / z;
        ent_sum -= pc * std::log(pc + 1e-12);
      }
      auto clampv = [](int v) { return std::clamp(v, -500, 500); };
      regret += clampv(ch[best].score) - clampv(ch[pred].score);
      const int reg = clampv(ch[best].score) - clampv(ch[pred].score);
      tie10 += reg <= 10;   // as Experimental.py tie_tol
      blunder += reg > 50;  // as Experimental.py blunder_cp
      top1 += (pred == best) || (ch[pred].score == ch[best].score);
      int better = 0;
      const double lb = out[ch[best].enc];
      for (int c = 0; c < n; ++c)
        better += out[ch[c].enc] > lb;
      top3 += better < 3;
      ++rows;
    }
    std::cout << "policy_eval rows " << rows << " top1(tie-aware) "
              << double(top1) / rows << " top3 " << double(top3) / rows
              << " ce_hard " << ce / rows << " regret " << regret / rows
              << " top1_tie10 " << double(tie10) / rows
              << " blunder50 " << double(blunder) / rows
              << " pmax " << pmax_sum / rows << " entropy " << ent_sum / rows
              << std::endl;
    return 0;
  }
  if (parser.has_option("log_evals")) {
    set_eval_log(parser.as<std::string>("log_evals"),
                 parser.has_option("log_rate") ? parser.as<int>("log_rate")
                                               : 1000);
  }
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
    max_nodes = std::stoull(parser.as<std::string>("nodes"));
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
    // --label_diag MS: per position one "diag" line (search bookkeeping of the
    // normal root search, then a reference search of MS ms with a cleared TT)
    const int label_diag_ms =
        parser.has_option("label_diag") ? parser.as<int>("label_diag") : 0;
    const int child_time =
        parser.has_option("child-time") ? parser.as<int>("child-time") : time;
    // Node budget instead of a time limit (reproducible labels, independent of
    // the CPU load): --nodes N for every search, or --nodes_phase A,B,C,D for
    // positions with >= 17, 13-16, 9-12 and <= 8 pieces (root and child searches
    // alike, by the searched position's piece count). The time limit then only
    // serves as a safety net.
    std::array<size_t, 4> phase_nodes;
    phase_nodes.fill(max_nodes);
    if (parser.has_option("nodes_phase")) {
      std::stringstream ss(parser.as<std::string>("nodes_phase"));
      std::string tok;
      for (int i = 0; i < 4 && std::getline(ss, tok, ','); ++i)
        phase_nodes[i] = std::stoull(tok);
    }
    const bool node_budget =
        parser.has_option("nodes") || parser.has_option("nodes_phase");
    auto budget = [&](const Position &p) {
      const int pc = p.piece_count();
      return phase_nodes[pc >= 17 ? 0 : pc >= 13 ? 1 : pc >= 9 ? 2 : 3];
    };
    const int root_time = node_budget ? 10000 : time;
    const int child_time_eff = node_budget ? 10000 : child_time;
    ROOT_FAIL_LOW_BOUND = true; // labelling only, see types.h

    TT.resize_in_mb(hash_size);
    // --no_mlh: the root value is searched without MLH (value label), child
    // searches with MLH (policy labels). Confirmed to the rescorer (Training/Rust
    // Data.rs), which refuses engines that do not print it.
    if (no_mlh)
      std::cout << "no_mlh" << std::endl;
    std::string current;
    while (std::getline(std::cin, current)) {
      if (current == "terminate") {
        std::exit(-1);
      }

      TT.age_counter++;
      const auto pos = Position::pos_from_fen(current);

      set_mlh(!no_mlh);
      root_diag = RootDiag{};
      if (label_diag_ms > 0)
        TT.clear(); // no entries of the previous (deep) reference search
      const Value root_value = search_value(pos, root_time, budget(pos));
      if (label_diag_ms > 0) {
        // label diagnostics: root bookkeeping of this search + a deeper reference
        const RootDiag d = root_diag;
        TT.clear();
        const Value ref = search_value(pos, label_diag_ms, SIZE_MAX);
        const RootDiag r = root_diag;
        std::cout << "diag " << pos.piece_count() << " " << d.nodes << " "
                  << d.completed_depth << " " << d.completed_score << " "
                  << d.interrupted << " " << d.partial_used << " " << d.partial_lower
                  << " " << d.fail_lows << " " << d.fail_highs << " " << root_value
                  << " " << ref << " " << r.completed_depth << " " << d.fail_low_score
                  << std::endl;
        set_mlh(true);
        continue;
      }
      set_mlh(true);

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
        const Value q = -search_value(child, child_time_eff, budget(child));

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

    // --- position sampling (see Reservoir in GameLogic.h) ---
    //   --sample_prob P     keep each new search-tree position with prob. P
    //   --sample_cap N      at most N positions per game (reservoir)
    //   --sample_bands a,b,c,d  extra keep prob. by |static eval| band
    //                       (<=50, 51-200, 201-400, >400); default 1,1,1,1
    // Budget per move: --nodes N (x2 for a 2-PV move) and/or --time MS; with
    // --nodes and no --time the search is limited by nodes only.
    reservoir.enabled = true;
    reservoir.generator.seed(adj_seed ^ 0xD1B54A32D192ED03ull);
    if (parser.has_option("sample_prob"))
      reservoir.sample_prob =
          std::stod(parser.as<std::string>("sample_prob"));
    if (parser.has_option("sample_cap"))
      reservoir.max_samples = parser.as<int>("sample_cap");
    if (parser.has_option("sample_bands")) {
      std::stringstream ss(parser.as<std::string>("sample_bands"));
      std::string item;
      for (size_t b = 0; b < reservoir.band_accept.size(); ++b) {
        if (!std::getline(ss, item, ','))
          throw std::runtime_error("--sample_bands needs 4 values");
        reservoir.band_accept[b] = std::stod(item);
      }
    }
    const bool node_budget = parser.has_option("nodes");
    if (node_budget && !parser.has_option("time"))
      time = 1000000;
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

        auto root_moves = searchValueMultiPV(
            board, this_move_pv, depth, this_move_pv * time,
            node_budget ? this_move_pv * max_nodes : max_nodes, false,
            std::cout);

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
