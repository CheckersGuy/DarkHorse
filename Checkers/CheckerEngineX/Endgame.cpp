#include "Endgame.h"

void TableBase::load_table_base(std::string path) {
  int i, status, nerrors;
  int max_pieces;
  /* Check that db files are present, get db type and size. */

  status = egdb_identify(path.c_str(), &egdb_type, &max_pieces);
  // std::cout << "MAX_PIECES: " << max_pieces << std::endl;
  if (max_pieces < num_pieces) {
    std::cout << "Can not handle that many pieces" << std::endl;
    std::exit(-1);
  }
  if (status) {
    // printf("No database found at %s\n", DB_PATH);
    std::exit(-1);
  }
  // printf("Database type %d found with max pieces %d\n", egdb_type,
  // max_pieces);

  /* Open database for probing. */
  handle =
      egdb_open(EGDB_NORMAL, num_pieces, 1000, path.c_str(), [](char *msg) {});
  if (!handle) {
    printf("Error returned from egdb_open()\n");
    std::exit(-1);
  }
}

TB_RESULT TableBase::probe(Position pos) {
  if (pos.has_jumps() || pos.piece_count() > num_pieces) {
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
}
