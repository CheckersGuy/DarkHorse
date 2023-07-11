//
// Created by leagu on 13.09.2021.
//
#include "../../Training/BloomFilter.h"
#include "../../Training/Sample.h"
#include "../../Training/Util/Compress.h"
#include "CmdParser.h"
#include "MGenerator.h"
#include "egdb.h"
#include <assert.h>
#include <chrono>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>

#define DB_PATH "E:\kr_english_wld"

void print_msgs(char *msg) { printf("%s", msg); }

Result get_tb_result(Position pos, int max_pieces, EGDB_DRIVER *handle) {
  if (pos.has_jumps() || Bits::pop_count(pos.BP | pos.WP) > max_pieces)
    return UNKNOWN;

  EGDB_NORMAL_BITBOARD board;
  board.white = pos.WP;
  board.black = pos.BP;
  board.king = pos.K;

  EGDB_BITBOARD normal;
  normal.normal = board;
  auto val = handle->lookup(
      handle, &normal, ((pos.color == BLACK) ? EGDB_BLACK : EGDB_WHITE), 0);

  if (val == EGDB_UNKNOWN)
    return UNKNOWN;

  if (val == EGDB_WIN)
    return (pos.color == BLACK) ? BLACK_WON : WHITE_WON;

  if (val == EGDB_LOSS)
    return (pos.color == BLACK) ? WHITE_WON : BLACK_WON;

  if (val == EGDB_DRAW)
    return DRAW;

  return UNKNOWN;
}

int main(int argl, const char **argc) {

  int i, status, max_pieces, nerrors;
  EGDB_TYPE egdb_type;
  EGDB_DRIVER *handle;

  /* Check that db files are present, get db type and size. */
  status = egdb_identify(DB_PATH, &egdb_type, &max_pieces);
  std::cout << "MAX_PIECES: " << max_pieces << std::endl;

  if (status) {
    printf("No database found at %s\n", DB_PATH);
    return (1);
  }
  printf("Database type %d found with max pieces %d\n", egdb_type, max_pieces);

  /* Open database for probing. */
  handle = egdb_open(EGDB_NORMAL, max_pieces, 4000, DB_PATH, print_msgs);
  if (!handle) {
    printf("Error returned from egdb_open()\n");
    return (1);
  }
  std::cout << "Starting Rescoring the training data" << std::endl;
  std::string in_file("../Training/TrainData/reinf.train");
  std::string out_file("../Training/TrainData/reinfformatted.train");

  create_samples_from_games(in_file, out_file, max_pieces, handle);
  std::cout << "Done rescoring" << std::endl;
  handle->close(handle);

  return 0;
}
