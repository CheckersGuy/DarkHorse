#include "helper.h"

extern "C" void load(char *path, int cache_size, int num_pieces) {
  base.cache_size = cache_size;
  base.num_pieces = num_pieces;
  base.load_table_base(path);
}

extern "C" int probe(char *fen_string) {
  return static_cast<int>(base.probe(Position::pos_from_fen(fen_string)));
}

extern "C" void print_fen(char *fen_string) {
  Position::pos_from_fen(fen_string).print_position();
}
