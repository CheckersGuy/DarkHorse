#ifndef PARAMS_H
#define PARAMS_H

// Central list of tunable search parameters (for SPSA).
//   MainEngine --list_params        prints name,value,min,max,c_end per line
//   MainEngine --RFP_BASE 22 ...    overrides parameters (clamped to range)
// The match tool passes them per engine via --opts1 RFP_BASE=22.

#include "types.h"
#include <algorithm>
#include <cstring>
#include <ostream>
#include <vector>

struct TuneParam {
  const char *name;
  int *value;
  int min, max;
  double c_end; // SPSA perturbation size at the end of tuning
};

inline std::vector<TuneParam> &tune_params() {
  static std::vector<TuneParam> params = {
      {"RFP_BASE", &RFP_BASE, 0, 80, 4},
      {"RFP_SLOPE", &RFP_SLOPE, 10, 60, 3},
      {"RFP_IMPROVING", &RFP_IMPROVING, 0, 20, 1.5},
      {"ASP_FH_RED_FP", &ASP_FH_RED_FP, 0, 3072, 128},
      {"ALPHA_RAISE_RED_FP", &ALPHA_RAISE_RED_FP, 0, 4096, 128},
      {"ALPHA_RAISE_MIN_D", &ALPHA_RAISE_MIN_D, 1, 6, 0.7},
      {"ALPHA_RAISE_MAX_D", &ALPHA_RAISE_MAX_D, 6, 30, 1.5},
      {"RFP_BLEND", &RFP_BLEND, 0, 1, 0},
      {"RFP_BLEND_W", &RFP_BLEND_W, 0, 1024, 40},
      {"TT_SHALLOW_MARGIN", &TT_SHALLOW_MARGIN, 0, 300, 8},
      {"QS_BLEND", &QS_BLEND, 0, 1, 0},
      {"QS_BLEND_W", &QS_BLEND_W, 0, 1024, 40},
      {"TTPV_PROPAGATE", &TTPV_PROPAGATE, 0, 1, 0},
      {"IMPROVING_BETA", &IMPROVING_BETA, 0, 1, 0},
      {"SE_NEG_EXT_FP", &SE_NEG_EXT_FP, 0, 4096, 128},
      {"SE_NEG_CUTNODE", &SE_NEG_CUTNODE, 0, 1, 0},
      {"SE_NEG_CUT_EXT_FP", &SE_NEG_CUT_EXT_FP, 0, 4096, 128},
      {"SE_EXT_FP", &SE_EXT_FP, 0, 2048, 128},
      {"SE_DOUBLE_EXT_FP", &SE_DOUBLE_EXT_FP, 1024, 3072, 128},
      {"SINGLE_EXT_FP", &SINGLE_EXT_FP, 0, 2048, 128},
      {"PV_CAPTURE_EXT_FP", &PV_CAPTURE_EXT_FP, 0, 2048, 128},
      {"RECAPTURE_EXT_FP", &RECAPTURE_EXT_FP, 0, 2048, 128},
      {"PROBCUT_IMP_RED_FP", &PROBCUT_IMP_RED_FP, 0, 3072, 128},
      {"LMR_CARRY_PCT", &LMR_CARRY_PCT, 0, 100, 8},
      {"POLICY_SCALE_X100", &POLICY_SCALE_X100, 50, 500, 15},
      // net-specific scale: tuned alone per new value net (spsa.py --only), excluded
      // from normal SPSA runs (spsa.py SCALE_PARAMS)
      {"VALUE_SCALE_X100", &VALUE_SCALE_X100, 50, 200, 4},
      {"RAZOR_BASE", &RAZOR_BASE, 50, 300, 15},
      {"RAZOR_QUAD", &RAZOR_QUAD, 2, 30, 1.5},
      {"PROBCUT_MARGIN", &prob_cut, 5, 80, 4},
      {"PROBCUT_IMPROVING", &PROBCUT_IMPROVING, 0, 30, 2},
      {"SE_MARGIN", &SE_MARGIN, 1, 6, 0.5},
      {"SE_DOUBLE_MARGIN", &SE_DOUBLE_MARGIN, 5, 80, 4},
      {"LMR_BASE_X100", &LMR_BASE_X100, 0, 150, 8},
      {"LMR_FACTOR_X100", &LMR_FACTOR_X100, 30, 130, 6},
      {"LMR_PROB_FLOOR_X1000", &LMR_PROB_FLOOR_X1000, 5, 100, 4},
      {"LMP_TAIL_1", &LMP_TAIL_1, 1, 300, 5},
      {"LMP_TAIL_2", &LMP_TAIL_2, 1, 300, 5},
      {"LMP_TAIL_3", &LMP_TAIL_3, 1, 300, 5},
      {"LMP_TAIL_4", &LMP_TAIL_4, 1, 300, 5},
      {"LMP_TAIL_5", &LMP_TAIL_5, 1, 300, 5},
      {"LMP_TAIL_6", &LMP_TAIL_6, 1, 300, 5},
      {"ASP_WINDOW", &asp_wind, 5, 40, 2},
      {"ASP_DIV", &ASP_DIV, 2, 20, 1},
      {"ASP_THRESHOLD", &ASP_THRESHOLD, 100, 500, 20},
      {"ASP_GROWTH_X100", &ASP_GROWTH_X100, 110, 250, 10},
      {"MLH_RAMP_LO", &MLH_RAMP_LO, 150, 450, 15},
      {"MLH_BONUS", &MLH_BONUS, 200, 450, 15},
      {"PROBCUT_MIN_DEPTH", &PROBCUT_MIN_DEPTH, 3, 10, 0.8},
      {"PROBCUT_RED_FP", &PROBCUT_RED_FP, 2048, 6144, 256},
      {"SE_TT_DEPTH_MARGIN", &SE_TT_DEPTH_MARGIN, 1, 8, 0.8},
      {"SE_DOUBLE_PLY_X10", &SE_DOUBLE_PLY_X10, 10, 40, 2},
      {"TT_DEPTH_WEIGHT", &TT_DEPTH_WEIGHT, 1, 15, 1},
      {"TT_AGE_WEIGHT", &TT_AGE_WEIGHT, 1, 20, 1},
      {"TT_OVERWRITE_MARGIN", &TT_OVERWRITE_MARGIN, 0, 10, 0.8},
      {"LMR_TABLE_MAX", &LMR_TABLE_MAX, 2, 8, 0.6},
      {"LMR_PV_FP", &LMR_PV_FP, 0, 3072, 128},
      {"LMR_NOT_IMPROVING_FP", &LMR_NOT_IMPROVING_FP, 0, 3072, 128},
      {"LMR_TTPV_FP", &LMR_TTPV_FP, 0, 3072, 128},
      {"LMR_TTPV_ALPHA_FP", &LMR_TTPV_ALPHA_FP, 0, 3072, 128},
      {"LMR_TTPV_DEPTH_FP", &LMR_TTPV_DEPTH_FP, 0, 3072, 128},
      {"LMR_CUTNODE_FP", &LMR_CUTNODE_FP, 0, 3072, 128},
      // depth limits: 1000 = unlimited (old behaviour). c_end 0 = listed for
      // --NAME but not tuned by spsa.py; test concrete values with SPRT.
      {"RFP_MAX_DEPTH", &RFP_MAX_DEPTH, 1, 1000, 0},
      // switches under test (types.h), not tuned by SPSA
      {"PC_EXCL_FIX", &PC_EXCL_FIX, 0, 1, 0},
      {"LMP1_COUNT", &LMP1_COUNT, 0, 40, 0},
      {"TT_CAPTURES", &TT_CAPTURES, 0, 1, 0},
      {"RAZOR_BLEND_W", &RAZOR_BLEND_W, 0, 1024, 0},
      {"ENDGAME_PIECES", &ENDGAME_PIECES, 2, 24, 0},
      {"ENDGAME_LMR_FP", &ENDGAME_LMR_FP, 0, 3072, 0},
      {"FORCING_LMP", &FORCING_LMP, 0, 1, 0},
      {"QS_THREAT_PLIES", &QS_THREAT_PLIES, 0, 8, 0},
      {"PROMO_EXT_FP", &PROMO_EXT_FP, 0, 2048, 0},
      {"TWO_REPLY_EXT_FP", &TWO_REPLY_EXT_FP, 0, 2048, 0},
      {"SE_SUBTREE", &SE_SUBTREE, 0, 1, 0},
      {"RAZOR_MAX_DEPTH", &RAZOR_MAX_DEPTH, 1, 1000, 0},
  };
  return params;
}

inline bool &lmp_tails_set() {
  static bool set = false;
  return set;
}

// Sets a parameter by name (clamped to its range); false if unknown.
inline bool set_param(const char *name, int value) {
  for (auto &p : tune_params()) {
    if (std::strcmp(p.name, name) == 0) {
      *p.value = std::clamp(value, p.min, p.max);
      if (std::strncmp(name, "LMP_TAIL_", 9) == 0)
        lmp_tails_set() = true;
      return true;
    }
  }
  return false;
}

// Recomputes tables derived from parameters. Call after set_param.
inline void apply_params() {
  POLICY_SCALE = POLICY_SCALE_X100 / 100.0f;
  LMR_TABLE_2D =
      compute_lmr_table(LMR_BASE_X100, LMR_FACTOR_X100, LMR_TABLE_MAX);
  if (lmp_tails_set()) {
    const int tails[9] = {0,          LMP_TAIL_1, LMP_TAIL_2,
                          LMP_TAIL_3, LMP_TAIL_4, LMP_TAIL_5,
                          LMP_TAIL_6, LMP_TAIL_6, LMP_TAIL_6};
    LMP_MASS[0] = 1.0f; // depth 0 never prunes
    for (int d = 1; d < 9; ++d)
      LMP_MASS[d] = 1.0f - tails[d] / 10000.0f;
  }
}

inline void print_params(std::ostream &out) {
  for (auto &p : tune_params())
    out << p.name << "," << *p.value << "," << p.min << "," << p.max << ","
        << p.c_end << "\n";
}

#endif
