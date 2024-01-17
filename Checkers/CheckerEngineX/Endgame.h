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
  EGDB_DRIVER *mtc_handle;
  int num_pieces{8}; // only used for the wdl-tablebase
  uint64_t cache_size{500};

  ~TableBase();

  void load_table_base(std::string path);

  TB_RESULT probe(Position pos);

  int get_num_pieces();

  void close();
};
