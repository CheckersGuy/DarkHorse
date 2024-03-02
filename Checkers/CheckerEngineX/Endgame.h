#include "MGenerator.h"
#include "Move.h"
#include "Position.h"
#include "egdb.h"
#include "types.h"
#include <cstdint>
#include <optional>
#include <stdio.h>
#include <string>
struct TableBase {
  EGDB_DRIVER *handle;
  EGDB_DRIVER *dtw_handle;
  EGDB_DRIVER *mtc_handle;
  int num_pieces{6}; // only used for the wdl-tablebase
  uint64_t cache_size{500};

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
