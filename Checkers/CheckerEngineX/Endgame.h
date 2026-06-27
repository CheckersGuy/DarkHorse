#include "MGenerator.h"
#include "Move.h"
#include "Position.h"

#ifdef _WIN32
#include "egdb.h"
#endif
#include "egdb.h"
#include "types.h"
#include <cstdint>
#include <optional>
#include <stdio.h>
#include <string>
#include <unordered_map>

struct TableBase {

  int num_pieces = 0;
  EGDB_DRIVER *handle;
  EGDB_DRIVER *dtw_handle;
  EGDB_DRIVER *mtc_handle;
  uint64_t cache_size{2000};
#ifdef CHECKERBOARD
  char *reply;
#endif
  ~TableBase();

  void load_table_base(std::string path, int num_pieces);

  void load_dtw_base(std::string path, int num_pieces);

  void load_mtc_base(std::string path, int num_pieces);

  TB_RESULT probe(Position pos);

  std::optional<int> probe_dtw(Position pos);

  std::optional<int> probe_mtc(Position pos);

  int get_num_pieces();

  std::optional<Move> find_best_mtc(Position pos);

  void close();
};
