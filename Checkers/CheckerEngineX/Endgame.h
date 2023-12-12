#include <stdio.h>
#include <string>

enum TB_RESUlT { WIN, LOSS, DRAW, UNKNOWN };

struct TableBase {

  void load_table_base(std::string path);

  TB_RESUlT probe_table_base();

  int get_num_pieces();

  void close();
};
