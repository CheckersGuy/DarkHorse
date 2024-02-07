#include "helper.h"

extern "C" void load(char *path, int cache_size, int num_pieces) {
  // would need seperate caches for both
  base.cache_size = cache_size;
  base.num_pieces = num_pieces;
  base.load_table_base(path);
}

extern "C" void load_dtw(char *dtw_path, int cache_size, int num_pieces) {
  base.cache_size = cache_size;
  base.num_pieces = num_pieces;
  base.load_dtw_base(dtw_path);
}

extern "C" int probe(char *fen_string) {
  return static_cast<int>(base.probe(Position::pos_from_fen(fen_string)));
}

extern "C" int probe_dtw(char *fen_string) {
  auto dtw = base.probe_dtw(Position::pos_from_fen(fen_string));
  if (dtw.has_value()) {
    return dtw.value();
  }
  return -1000;
}

extern "C" void print_fen(char *fen_string) {
  Position::pos_from_fen(fen_string).print_position();
}

extern "C" int move_played(char *orig, char *next) {
  Position o = Position::pos_from_fen(orig);
  Position n = Position::pos_from_fen(orig);

  if (o.color == BLACK) {
    o = o.get_color_flip();
  }
  if (n.color == WHITE) {
    n = n.get_color_flip();
  }

  auto result = o.get_move(o, n);
  if (result.has_value()) {
    result.value().get_move_encoding();
  }

  return -1;
}
