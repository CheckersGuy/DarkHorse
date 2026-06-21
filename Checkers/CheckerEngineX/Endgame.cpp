#include "Endgame.h"
#include "MGenerator.h"
#include "egdb.h"
#include "types.h"
#include <chrono>
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
  std::cout << "Loaded MTC with" << num_pieces << " pieces" << std::endl;
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

  return val;
}

// needs to be reworked before working on the other functions
// see claude code for more info
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
    TBConversionResult r{
        false, 0, std::nullopt}; // no legal moves -> immediate, 0-ply loss
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
  Move best_move;
  for (int i = 0; i < liste.length(); ++i) {
    Position child = pos;
    child.make_move(liste[i]);

    auto wdl_probe = base.probe(pos);
    if (wdl_probe == TB_RESULT::DRAW) {
      continue;
    }

    auto child_result = solve_mtc(false, child, budget - 1);
    if (!child_result.has_value()) {
      any_unresolved = true;
      continue;
    }

    const int total_plies = child_result->plies + 1;

    if (!child_result->is_winning) {
      found_win = true;
      best_win_plies = std::min(best_win_plies, total_plies);
      best_move = liste[i];
    } else {
      best_loss_plies = std::max(best_loss_plies, total_plies);
      best_move = liste[i];
    }
  }

  if (found_win) {
    TBConversionResult r{true, best_win_plies, best_move};
    // proven.emplace(pos, r);
    return r;
  }
  if (any_unresolved) {
    return std::nullopt; // can't rule out a draw, or a win needing more depth
  }
  if (best_loss_plies >= 0) {

    TBConversionResult r{false, best_loss_plies, best_move};
    // proven.emplace(pos, r);
    return r;
  }
  return std::nullopt;
}

// TO BE WORKED ON
std::optional<Move> Solver::find_best_mtc(Position pos, int budget) {
  auto wdl = probe(pos);
  if (wdl == TB_RESULT::DRAW || wdl == TB_RESULT::UNKNOWN) {
    return std::nullopt;
  }
  MoveListe liste;
  get_moves(pos, liste);

  std::optional<Move> best;
  int best_mtc = std::numeric_limits<int>::max();

  for (int i = 0; i < liste.length(); ++i) {
    Position child = pos;
    child.make_move(liste[i]);

    auto child_wdl = probe(child);
    if (child_wdl == TB_RESULT::DRAW) {
      continue; // only moves that keep the opponent lost are candidates
    }

    auto mtc = solve_mtc(pos, budget);
    if (!mtc.has_value()) {
      continue;
    }

    if (mtc.value() < best_mtc) {
      best_mtc = mtc.value();
      best = liste[i];
    }
  }
}

std::vector<Move> Solver::playout_mtc(Position pos, int budget, int max_moves) {
  std::vector<Move> moves;
  for (auto i = 0; i < max_moves; ++i) {

    auto solve_probe = solve_mtc(pos, budget);
    if (!solve_probe.has_value()) {
      break;
    }
    if (!solve_probe->move.has_value()) {
      std::cout << "Did not find a move" << std::endl;

      MoveListe liste;
      get_moves(pos, liste);

      break;
    }

    const auto move = solve_probe->move.value();

    moves.emplace_back(move);
    pos.make_move(move);
    std::cout << "IsWinning: " << solve_probe->is_winning
              << " Plies: " << solve_probe->plies << std::endl;
  }
  return moves;
}
