#include "Endgame.h"
#include "MGenerator.h"
#include "egdb.h"
#include "types.h"
#include <optional>

TableBase::~TableBase() {}
// change how loading works when num_pieces > max_pieces
// just default to max_pieces in that case !
//

void TableBase::load_table_base(std::string path) {
  int i, status, nerrors;
  int max_pieces;

  EGDB_TYPE egdb_type;
  /* Check that db files are present, get db type and size. */

  status = egdb_identify(path.c_str(), &egdb_type, &max_pieces);
  // std::cout << "MAX_PIECES: " << max_pieces << std::endl;

  num_pieces = std::min(num_pieces, max_pieces);

  if (status) {
    printf("No database found at %s\n", path.c_str());
    std::exit(-1);
  }

  handle = egdb_open(EGDB_NORMAL, num_pieces, cache_size, path.c_str(),
                     [](char *msg) {

                     });

  std::cout << "Loaded WDL with" << num_pieces << " pieces" << std::endl;

  if (!handle) {
    write_to_logfile("Could not load the tablebase");
    std::cerr << "Error returned from egdb_open()" << std::endl;
    std::exit(-1);
  }
}

void TableBase::load_dtw_base(std::string path) {
  int i, status, nerrors;
  int max_pieces;

  EGDB_TYPE egdb_type;
  /* Check that db files are present, get db type and size. */

  status = egdb_identify(path.c_str(), &egdb_type, &max_pieces);
  // std::cout << "MAX_PIECES: " << max_pieces << std::endl;

  num_pieces = std::min(num_pieces, max_pieces);

  if (status) {
    printf("No database found at %s\n", path.c_str());
    std::exit(-1);
  }

  dtw_handle = egdb_open(EGDB_NORMAL, num_pieces, cache_size, path.c_str(),
                         [](char *msg) {});
  std::cout << "Loaded DTW with" << num_pieces << " pieces" << std::endl;
  if (!dtw_handle) {
    std::cerr << "Error returned from egdb_open()" << std::endl;
    std::exit(-1);
  }
}

void TableBase::load_mtc_base(std::string path) {
  int i, status, nerrors;
  int max_pieces;

  EGDB_TYPE egdb_type;
  /* Check that db files are present, get db type and size. */

  status = egdb_identify(path.c_str(), &egdb_type, &max_pieces);
  // std::cout << "MAX_PIECES: " << max_pieces << std::endl;

  num_pieces = std::min(num_pieces, max_pieces);

  if (status) {
    printf("No database found at %s\n", path.c_str());
    std::exit(-1);
  }

  mtc_handle = egdb_open(EGDB_NORMAL, num_pieces, cache_size, path.c_str(),
                         [](char *msg) {});
  std::cout << "Loaded DTW with" << num_pieces << " pieces" << std::endl;
  if (!mtc_handle) {
    std::cerr << "Error returned from egdb_open()" << std::endl;
    std::exit(-1);
  }
}

TB_RESULT TableBase::probe(Position pos) {
  // the kingsrow wld database does not have a valid value for any positions
  // where side-to-move can capture the kingsrow database only has valid
  // values for positions with atmost 5 pieces on one side
  if (pos.has_jumps() || pos.piece_count() > num_pieces ||
      Bits::pop_count(pos.BP) > 5 || Bits::pop_count(pos.WP) > 5) {
    return TB_RESULT::UNKNOWN;
  }
  if (handle == nullptr) {
    return TB_RESULT::UNKNOWN;
  }

  EGDB_NORMAL_BITBOARD board;
  board.white = pos.WP;
  board.black = pos.BP;
  board.king = pos.K;

  EGDB_BITBOARD normal;
  normal.normal = board;
  auto val = handle->lookup(
      handle, &normal, ((pos.color == BLACK) ? EGDB_BLACK : EGDB_WHITE), 0);

  if (val == EGDB_UNKNOWN)
    return TB_RESULT::UNKNOWN;

  if (val == EGDB_WIN)
    return TB_RESULT::WIN;

  if (val == EGDB_LOSS)
    return TB_RESULT::LOSS;

  if (val == EGDB_DRAW)
    return TB_RESULT::DRAW;

  return TB_RESULT::UNKNOWN;
}

std::optional<int> TableBase::probe_dtw(Position pos) {
  if (dtw_handle == nullptr) {
    return std::nullopt;
  }

  if (handle == nullptr) {
    return std::nullopt;
  }

  auto wdl = probe(pos);
  if (wdl != TB_RESULT::WIN && wdl != TB_RESULT::LOSS) {
    return std::nullopt;
  }

  EGDB_NORMAL_BITBOARD board;
  board.white = pos.WP;
  board.black = pos.BP;
  board.king = pos.K;

  EGDB_BITBOARD normal;
  normal.normal = board;
  auto val = dtw_handle->lookup(
      dtw_handle, &normal, ((pos.color == BLACK) ? EGDB_BLACK : EGDB_WHITE), 0);

  if (val > 0) {
    if (wdl == TB_RESULT::WIN) {
      return 2 * val + 1;
    } else {
      return 2 * val;
    }
  }
  return std::nullopt;
}

std::optional<int> TableBase::probe_mtc(Position pos) {

  if (pos.has_jumps() || pos.piece_count() > num_pieces ||
      Bits::pop_count(pos.BP) > 5 || Bits::pop_count(pos.WP) > 5) {
    return std::nullopt;
  }
  if (mtc_handle == nullptr) {
    return std::nullopt;
  }

  EGDB_NORMAL_BITBOARD board;
  board.white = pos.WP;
  board.black = pos.BP;
  board.king = pos.K;

  EGDB_BITBOARD normal;
  normal.normal = board;
  auto val = mtc_handle->lookup(
      mtc_handle, &normal, ((pos.color == BLACK) ? EGDB_BLACK : EGDB_WHITE), 0);

  if (val == EGDB_UNKNOWN) {
    return std::nullopt;
  }
}

std::optional<TBConversionResult> Solver::solve_mtc(Position pos, int budget) {

  auto wdl_probe = base.probe(pos);

  if (wdl_probe == TB_RESULT::WIN || wdl_probe == TB_RESULT::LOSS) {

    auto mtc_probe = base.probe_mtc(pos);

    if (mtc_probe.has_value()) {
      return TBConversionResult{wdl_probe == TB_RESULT::WIN, mtc_probe.value()};
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
