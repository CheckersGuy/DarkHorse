#include "MGenerator.h"
#include "Move.h"
#include "Position.h"
#include "egdb.h"
#include "types.h"
#include <cstdint>
#include <optional>
#include <stdio.h>
#include <string>
#include <unordered_map>
struct TableBase {

  EGDB_DRIVER *handle;
  EGDB_DRIVER *dtw_handle;
  EGDB_DRIVER *mtc_handle;
  int num_pieces{6};
  uint64_t cache_size{2000};

  ~TableBase();

  void load_table_base(std::string path);

  void load_dtw_base(std::string path);

  void load_mtc_base(std::string path);

  TB_RESULT probe(Position pos);

  std::optional<int> probe_dtw(Position pos);

  std::optional<int> probe_mtc(Position pos);

  int get_num_pieces();

  std::optional<Move> find_best_mtc(Position pos);

  void close();
};

struct TBConversionResult {
  bool is_winning; // true if the position is a win for the player to move
  int plies;       // the number of plies to the next conversion
};

struct Solver {

  std::unordered_map<Position,
                     TBConversionResult>
      cache; // caches already proven positions that are
             // not in the mtc database

  TableBase &base;

  Solver(TableBase &base) : base(base) {} // constructor for Solver

  std::optional<TBConversionResult> solve_mtc(Position pos, int budget) {

    auto wdl_probe = base.probe(pos);

    if (wdl_probe == TB_RESULT::WIN || wdl_probe == TB_RESULT::LOSS) {

      auto mtc_probe = base.probe_mtc(pos);

      if (mtc_probe.has_value()) {
        return TBConversionResult{wdl_probe == TB_RESULT::WIN,
                                  mtc_probe.value()};
      }
    }

    MoveListe liste;
    get_moves(pos, liste);

    if (liste.length() == 0) {
      TBConversionResult r{false, 0}; // no legal moves -> immediate, 0-ply loss
      // proven.emplace(pos, r);
      return r;
    }

    if (budget <= 0) {
      return std::nullopt; // out of search depth -- do NOT cache this
    }

    bool any_unresolved = false;
    bool found_win = false;
    int best_win_plies = std::numeric_limits<int>::max();
    int best_loss_plies = -1;

    for (int i = 0; i < liste.length(); ++i) {
      Position child = pos;
      child.make_move(liste[i]);

      auto child_result = solve_mtc(child, budget - 1);
      if (!child_result.has_value()) {
        any_unresolved = true;
        continue;
      }

      const int total_plies = child_result->plies + 1;
      if (!child_result->is_winning) {
        found_win = true;
        best_win_plies = std::min(best_win_plies, total_plies);
      } else {
        best_loss_plies = std::max(best_loss_plies, total_plies);
      }
    }

    if (found_win) {
      TBConversionResult r{true, best_win_plies};
      // proven.emplace(pos, r);
      return r;
    }
    if (any_unresolved) {
      return std::nullopt; // can't rule out a draw, or a win needing more depth
    }
    if (best_loss_plies >= 0) {
      TBConversionResult r{false, best_loss_plies};
      // proven.emplace(pos, r);
      return r;
    }

    return std::nullopt;
  }
};
