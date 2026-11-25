//
// Created by Robin on 18.12.2017.
//

#ifndef CHECKERSTEST_TYPES_H
#define CHECKERSTEST_TYPES_H

#include <algorithm>
#include <array>
#include <assert.h>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <thread>
// Utility functions and other stuff

enum Result : uint8_t { BLACK_WON = 1, WHITE_WON = 2, DRAW = 3, UNKNOWN = 0 };

inline uint64_t getSystemTime() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

// overload trick

template <typename... Ts> struct overload : Ts... {
  using Ts::operator()...;
};
template <typename... Ts> overload(Ts...) -> overload<Ts...>;

template <size_t base>
auto power_lambda = [](size_t exp) {
  size_t result = 1;
  for (auto i = 0; i < exp; ++i) {
    result *= base;
  }
  return result;
};

template <size_t size, typename Generator>
constexpr auto get_lut(Generator &&generator) {
  using data_type = decltype(generator(0));
  std::array<data_type, size> result{};

  for (auto i = 0; i < size; ++i) {
    result[i] = generator(i);
  }
  return result;
}
constexpr std::array<int, 25> piece_count_to_bucket = {
    11, 11, 11, 11, 10, 9, 8, 7, 6, 5, 4, 3, 3,
    2,  2,  2,  1,  1,  1, 0, 0, 0, 0, 0, 0};
constexpr std::array<int, 32> BIT_TO_BOARD = {
    3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,  15, 14, 13, 12,
    19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28};
constexpr std::array<int, 32> BOARD_TO_BIT = {
    3,  2,  1,  0,  7,  6,  5,  4,  11, 10, 9,  8,  15, 14, 13, 12,
    19, 18, 17, 16, 23, 22, 21, 20, 27, 26, 25, 24, 31, 30, 29, 28};

constexpr uint32_t INNER_SQUARES = 132638688;
constexpr uint32_t OUTER_SQUARES = 135796752;
constexpr uint32_t MASK_L3 = 14737632u;
constexpr uint32_t MASK_L5 = 117901063u;
constexpr uint32_t MASK_R3 = 117901056u;
constexpr uint32_t MASK_R5 = 3772834016u;
constexpr uint32_t MASK_COL_1 = 286331153u;
constexpr uint32_t MASK_COL_2 = 572662306u;
constexpr uint32_t MASK_COL_3 = 1145324612u;
constexpr uint32_t MASK_COL_4 = 2290649224u;
constexpr uint32_t PROMO_SQUARES_WHITE = 0xfu;
constexpr uint32_t PROMO_SQUARES_BLACK = 0xf0000000u;
constexpr uint32_t OUTER_RING = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3) |
                                (1 << 4) | (1 << 12) | (1 << 20) | (1 << 28) |
                                (1 << 29) | (1 << 30) | (1 << 31) | (1 << 27) |
                                (1 << 19) | (1 << 11);
constexpr uint32_t CENTER = (1 << 9) | (1 << 10) | (1 << 13) | (1 << 14) |
                            (1 << 17) | (1 << 18) | (1 << 22) | (1 << 21);
constexpr uint32_t SINGLE_CORNER = (1 << 28) | (1 << 3);
constexpr uint32_t DOUBLE_CORNER = (1 << 0) | (1 << 4) | (1 << 31) | (1 << 27);
constexpr std::array<uint32_t, 8> MASK_ROWS = {
    0xf, 0xf << 4, 0xf << 8, 0xf << 12, 0xf << 16, 0xf << 20, 0xf << 24};

constexpr int LMR_MAX_DEPTH = 64;      // depths beyond this reuse the last row
constexpr int LMR_MAX_MOVE_INDEX = 40; // MoveListe holds at most 40 moves

// ---- tunable search parameters (see Params.h; --list_params, --NAME value)
// Scaled integers so SPSA can treat every parameter the same way.
inline int RFP_BASE = 20;      // RFP: eval - BASE - (SLOPE - IMPROVING*imp)*d
inline int RFP_SLOPE = 33;
inline int RFP_IMPROVING = 6;
// bundle of small search features (0 = off); SPRT H1 +4.3 on 2026-11-22
// depth changes in 1/1024 ply (fractional parts are carried down the line)
inline int ASP_FH_RED_FP = 824;      // root depth - this*fail_highs in aspiration
inline int ALPHA_RAISE_RED_FP = 2010; // depth - this after alpha is raised
inline int RFP_BLEND = 1;             // RFP returns (eval + beta) / 2
inline int TT_SHALLOW_MARGIN = 58;    // TT_LOWER cutoff at depth-1 if >= beta+this
// bundle C (0 = off); SPRT H1 +2.3 on 2026-11-22
inline int QS_BLEND = 1;       // qsearch fail-high blend
inline int TTPV_PROPAGATE = 1; // fail-low node inherits parent tt_pv
inline int IMPROVING_BETA = 1; // improving |= static_eval >= beta
// bundle E; SPRT H1 +5.2 on 2026-11-22 (old behaviour: 1, 0, 0, alpha-raise 1)
inline int SE_NEG_EXT_FP = 2049;    // negative singular extension
inline int SE_NEG_CUTNODE = 1;  // negative extension also at cut nodes
// SPSA-tunable constants of accepted features (defaults = tested behaviour)
inline int SE_NEG_CUT_EXT_FP = 1968; // negative extension at cut nodes
inline int SE_EXT_FP = 1114;         // singular extension
inline int SE_DOUBLE_EXT_FP = 1998;  // double extension
inline int SINGLE_EXT_FP = 1084;     // only one legal move
inline int PV_CAPTURE_EXT_FP = 940; // capture at a PV node
inline int RECAPTURE_EXT_FP = 962;  // capture after the opponent could jump
inline int RFP_BLEND_W = 466;       // RFP return: (W*eval + (1024-W)*beta)/1024
inline int QS_BLEND_W = 456;        // qsearch fail-high blend weight of best
inline int ALPHA_RAISE_MIN_D = 2;   // alpha-raise reduction for MIN < depth < MAX
inline int ALPHA_RAISE_MAX_D = 15;
inline int PROBCUT_IMP_RED_FP = 1008; // extra ProbCut reduction when improving
inline int LMR_CARRY_PCT = 39;    // % of the fractional LMR remainder carried (0 = off); NR-SPRT [-3,1] H1 +2.0 +- 3.9 (2026-11-22)
// material scaling of the net eval: eval * (BASE + SLOPE*(pieces-12)) / 1024
inline int RAZOR_BASE = 162;   // razoring: eval + BASE + QUAD*d*d <= alpha
inline int RAZOR_QUAD = 11;
inline int PROBCUT_IMPROVING = 12; // prob_beta = beta + prob_cut + this*imp
inline int LMR_BASE_X100 = 57;     // LMR: BASE/100 + FACTOR/100 * ln d * ln m
inline int LMR_FACTOR_X100 = 71;
inline int LMR_PROB_FLOOR_X1000 = 28; // policy probability floor in reduce()
inline int ASP_DIV = 8;               // aspiration: wind + (|s|-THRESHOLD)/DIV
inline int ASP_THRESHOLD = 282;       // aspiration widening starts here
inline int ASP_GROWTH_X100 = 156;     // window growth per re-search (x100)
inline int MLH_RAMP_LO = 268;         // eval blend with moves-left estimate
inline int MLH_BONUS = 267;           // bonus = t * max(0, MLH_BONUS - mlh)
// moves-left blend on/off. --eval-loop --no_mlh: the ROOT value is searched without
// MLH (value-net label = pure search value; the engine adds the MLH bonus itself),
// child searches (--children, policy labels) keep MLH: it separates the winning
// moves in decided positions.
inline bool USE_MLH = true;
// XORed into every TT key while MLH is off, so searches with and without MLH never
// share TT entries (their values differ)
inline uint64_t TT_KEY_SALT = 0;
// labelling (eval loop) only: when the search stops inside an iteration that has
// failed low and no root move beat alpha yet, return that fail-low score (an upper
// bound of the deeper value) instead of the older, more optimistic score of the last
// completed iteration. Measured on 6k positions vs a 50 ms reference: the failing-low
// cases (5 %) 86 -> 83 cp error, overall label MSE -1.6 %. Off in play.
inline bool ROOT_FAIL_LOW_BOUND = false;
constexpr uint64_t NO_MLH_TT_SALT = 0x9E3779B97F4A7C15ull;
inline void set_mlh(bool on) {
  USE_MLH = on;
  TT_KEY_SALT = on ? 0 : NO_MLH_TT_SALT;
}
inline int PROBCUT_MIN_DEPTH = 7;
inline int PROBCUT_RED_FP = 4093;     // ProbCut verification depth = d - this
inline int SE_TT_DEPTH_MARGIN = 2;    // SE needs tt depth >= depth - this
inline int SE_DOUBLE_PLY_X10 = 22;    // double ext. only while 10*ply < X10*rd
inline int RFP_MAX_DEPTH = 1000;      // 1000 = no limit (old behaviour)
inline int RAZOR_MAX_DEPTH = 1000;    // 1000 = no limit (old behaviour)
inline int TT_DEPTH_WEIGHT = 5;       // TT replacement: W_D*depth - W_A*age
inline int TT_AGE_WEIGHT = 7;
inline int TT_OVERWRITE_MARGIN = 4;   // same key overwritten if d+this > old

// LMR in fixed point (1/1024 ply), rounded towards zero at the end. The
// defaults (1024 = one ply) reproduce the old integer reductions exactly.
constexpr int LMR_ONE = 1024;
inline int LMR_TABLE_MAX = 6;          // cap of the table part (plies)
inline int LMR_PV_FP = 1014;           // less at PV nodes
inline int LMR_NOT_IMPROVING_FP = 1048;
inline int LMR_TTPV_FP = 882;         // tt_pv non-PV node: less by this
inline int LMR_TTPV_ALPHA_FP = 1076;   //   + this if tt value > alpha
inline int LMR_TTPV_DEPTH_FP = 924;   //   + this if tt depth >= depth
inline int LMR_CUTNODE_FP = 1097;      // more at cut nodes (not the tt move)

// table entries in 1/1024 ply: floor(raw * 1024), clamped to [0, max plies]
using LmrTable = std::array<std::array<int, LMR_MAX_MOVE_INDEX>, LMR_MAX_DEPTH>;
inline LmrTable compute_lmr_table(int base_x100, int factor_x100,
                                  int max_plies = 5) {
  LmrTable table{};
  const double base = base_x100 / 100.0, factor = factor_x100 / 100.0;
  for (int d = 0; d < LMR_MAX_DEPTH; ++d) {
    for (int m = 0; m < LMR_MAX_MOVE_INDEX; ++m) {
      const double depth = d + 1;
      const double move_index = m + 1;
      const double raw = base + factor * std::log(depth) * std::log(move_index);
      table[d][m] = std::clamp(static_cast<int>(raw * 1024.0), 0,
                               max_plies * 1024);
    }
  }
  return table;
}
inline LmrTable LMR_TABLE_2D = compute_lmr_table(57, 71, 6); // = LMR_BASE_X100, LMR_FACTOR_X100, LMR_TABLE_MAX

// Policy logits (out / 127) are multiplied by this before every softmax.
// > 1 sharpens the move probabilities used by LMP, LMR and futility pruning;
// the move ranking is unaffected. Set with --policy_scale.
inline float POLICY_SCALE = 1.0f;
// tunable form of POLICY_SCALE (x100); apply_params sets POLICY_SCALE from it,
// --policy_scale still overrides it afterwards
inline int POLICY_SCALE_X100 = 100;

// Time management (off = always search until the hard deadline, as before).
// With TM on, after each iteration the main thread predicts the duration of
// the next one (last iteration's duration * growth, growth = ratio of the last
// two iteration durations clamped to [TM_GROWTH_MIN, TM_GROWTH_MAX]) and does
// not start it if elapsed + prediction > budget * TM_STABILITY[stable], where
// 'stable' counts consecutive iterations with the same best move. The hard
// deadline stays the full budget. Set with --tm / CheckerBoard "set tm".
inline int TM_ENABLED = 0;
inline double TM_GROWTH_MIN = 1.2;
inline double TM_GROWTH_MAX = 3.0;
inline std::array<double, 5> TM_STABILITY = {1.0, 1.0, 0.9, 0.8, 0.7};
inline int TM_MIN_DEPTH = 6;

// LMP tails in units of 1e-4 (mass = 1 - tail/10000); depths 7-8 use TAIL_6.
// LMP_MASS is only recomputed from the tails when one of them is set
// (apply_params); the initialiser below uses the same formula.
inline int LMP_TAIL_1 = 99, LMP_TAIL_2 = 53, LMP_TAIL_3 = 40, LMP_TAIL_4 = 31,
           LMP_TAIL_5 = 20, LMP_TAIL_6 = 8;
inline std::array<float, 9> LMP_MASS = {
    1.0f,                    // depth 0 (unused, never prunes)
    1.0f - 99 / 10000.0f,    // depth 1  (same formula as apply_params)
    1.0f - 53 / 10000.0f,    // depth 2
    1.0f - 40 / 10000.0f,    // depth 3
    1.0f - 31 / 10000.0f,    // depth 4
    1.0f - 20 / 10000.0f,    // depth 5
    1.0f - 8 / 10000.0f,    // depth 6
    1.0f - 8 / 10000.0f,    // depth 7
    1.0f - 8 / 10000.0f     // depth 8
};

inline int SE_MAX_MULTI = 2;       // multi-extensions allowed along one line

inline int FP_MAX_DEPTH = 6;  // only prune at shallow remaining depth
inline int FP_BASE = 40;      // margin at lmr_depth 0
inline int FP_SLOPE = 35;     // extra margin per lmr_depth
inline int FP_MIN_PIECES = 8; // below this, king endgames: eval too unreliable
constexpr float FP_MIN_MASS = 0.5f; // policy mass that must be searched first

inline int SE_MIN_DEPTH = 6;
inline int SE_MARGIN = 2;         // s_beta = tt_value - SE_MARGIN * depth
inline int SE_DOUBLE_MARGIN = 23; // same scale as prob_cut

constexpr std::array<int, 9> LMP_COUNT = {0, 5, 7, 9, 13, 15, 18, 18, 18};

inline int prob_cut = 41; // tunable as PROBCUT_MARGIN

inline int asp_wind = 19; // tunable as ASP_WINDOW

constexpr int MAX_ASP = 200;
constexpr int CORRECTION_SIZE = 512;
constexpr int MAX_EVAL = 500;
constexpr uint64_t BLACK_RANDOM = 7985716234ull;
constexpr uint64_t singular_key = 311234512ull;

using Depth = int;
using Ply = int;
using Value = int;

enum SEARCH : int { MAX_PLY = 256 };
enum Score : int {
  SCORE_INFINITE = 15001,
  EVAL_INFINITE = 13000,
  MATE_IN_MAX_PLY = EVAL_INFINITE - MAX_PLY,
  MATED_IN_MAX_PLY = -MATE_IN_MAX_PLY,
  TB_WIN_MAX_PLY = 10000 - MAX_PLY,
  TB_LOSS_MAX_PLY = -10000 + MAX_PLY,
  TB_LOSS = -10000,
};

enum Color : int { BLACK = -1, WHITE = 1 };
enum PieceType : int {
  BPAWN = 0,
  WPAWN = 1,
  BKING = 2,
  WKING = 3,
  KING = 4,
  PAWN = 5,
  EMPTY = 6
};
enum Flag : uint8_t { None = 0u, TT_EXACT = 1u, TT_LOWER = 2u, TT_UPPER = 3u };

enum MoveType {
  PawnMove,
  KingMove,
  PromoMove,
  KingCapture,
  PawnCapture,
  PromoCapture
};
enum class TB_RESULT { WIN = 0, LOSS = 1, DRAW = 2, UNKNOWN = 3 };

inline void *std_aligned_alloc(size_t alignment, size_t size) {
#if defined(POSIXALIGNEDALLOC)
  void *mem;
  return posix_memalign(&mem, alignment, size) ? nullptr : mem;
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
  return _mm_malloc(size, alignment);
#elif defined(_WIN32)
  return _aligned_malloc(size, alignment);
#else
  return std::aligned_alloc(alignment, size);
#endif
}

inline void std_aligned_free(void *ptr) {

#if defined(POSIXALIGNEDALLOC)
  free(ptr);
#elif defined(_WIN32) && !defined(_M_ARM) && !defined(_M_ARM64)
  _mm_free(ptr);
#elif defined(_WIN32)
  _aligned_free(ptr);
#else
  free(ptr);
#endif
}

inline bool isEval(Value val) { return std::abs(val) < EVAL_INFINITE; }

inline bool isMateVal(Value val) { return std::abs(val) >= MATE_IN_MAX_PLY; }

inline Value loss(int ply) { return -EVAL_INFINITE + ply; }

inline Value tbloss(int ply) { return TB_LOSS + ply; };

constexpr Color operator~(Color color) { return static_cast<Color>(-color); }

inline bool isLoss(Value val) { return val <= TB_LOSS && isEval(val); }

inline bool isWin(Value val) { return val >= -TB_LOSS && isEval(val); }

inline bool isWinningEval(Value val) { return val >= MAX_EVAL; };
inline bool isLosingEval(Value val) { return val <= -MAX_EVAL; };
inline bool isDecesive(Value val) { return std::abs(val) >= MAX_EVAL; };

template <Color color> constexpr uint32_t defaultShift(const uint32_t maske) {
  if constexpr (color == BLACK) {
    return maske << 4u;
  } else {
    return maske >> 4u;
  }
}

template <Color color> constexpr uint32_t forwardMask(const uint32_t maske) {
  if constexpr (color == BLACK) {
    return ((maske & MASK_L3) << 3u) | ((maske & MASK_L5) << 5u);
  } else {
    return ((maske & MASK_R3) >> 3u) | ((maske & MASK_R5) >> 5u);
  }
}

template <Color color, PieceType type>
constexpr uint32_t get_neighbour_squares(uint32_t maske) {
  if constexpr (type == KING) {
    uint32_t squares = defaultShift<color>(maske) | forwardMask<color>(maske);
    squares |= forwardMask<~color>(maske) | defaultShift<~color>(maske);
    return squares;
  } else {
    return defaultShift<color>(maske) | forwardMask<color>(maske);
  }
}

inline void write_to_logfile(std::string msg) {
#ifdef _WIN32
  std::ofstream stream("E:\\logfile.txt", std::ios::out | std::ios::app);
#endif

#ifdef __linux__
  std::ofstream stream("/mnt/e/logfile.txt", std::ios::out | std::ios::app);
#endif
  stream << msg << std::endl;
  stream.flush();
}

// fast modulo operation for two 32-bit unsigned integers
inline uint32_t reduce(uint32_t a, uint32_t b) {
  const uint64_t op_a = static_cast<uint64_t>(a);
  const uint64_t op_b = static_cast<uint64_t>(b);
  const uint32_t result = static_cast<uint32_t>((op_a * op_b) >> 32);
  return result;
}

inline uint32_t round_nearest_power2(uint32_t v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;
  return v;
}

#endif
