#include "Position.h"
#include "egdb.h"
#include <stdio.h>
#include <string>
enum class TB_RESULT { WIN, LOSS, DRAW, UNKNOWN };

struct TableBase {
  EGDB_TYPE egdb_type;
  EGDB_DRIVER *handle;
  int num_pieces{6};
  void load_table_base(std::string path);

  TB_RESULT probe(Position pos);

  int get_num_pieces();

  void close();
};
