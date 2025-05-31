#include "helper.h"
#include "MGenerator.h"
#ifdef _WIN32
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

extern "C" int probe_with_position(unsigned int wp, unsigned int bp,
                                   unsigned int k, int color) {
  Position next = Position{};
  next.WP = wp;
  next.BP = bp;
  next.K = k;
  next.color = (color == -1) ? BLACK : WHITE;
  return static_cast<int>(base.probe(next));
}
extern "C" int probe_dtw_with_position(unsigned int wp, unsigned int bp,
                                       unsigned int k, int color) {
  Position next = Position{};
  next.WP = wp;
  next.BP = bp;
  next.K = k;
  next.color = (color == -1) ? BLACK : WHITE;

  auto dtw = base.probe_dtw(next);
  if (dtw.has_value()) {
    return dtw.value();
  }
  return -1000;
}
#endif
extern "C" void print_fen(char *fen_string) {
  Position::pos_from_fen(fen_string).print_position();
}

extern "C" int move_played(char *orig, char *next) {
  Position o = Position::pos_from_fen(orig);
  Position n = Position::pos_from_fen(next);
  auto result = o.get_move(o, n);
  if (result.has_value()) {
    if (o.color == Color::BLACK) {
      return result.value().flipped().get_move_encoding();
    }

    return result.value().get_move_encoding();
  }

  return -1;
}

extern "C" int move_played_pos(uint32_t o_wp, uint32_t o_bp, uint32_t o_k,
                               uint32_t n_wp, uint32_t n_bp, uint32_t n_k) {
  Position o = Position{};
  Position n = Position{};
  o.WP = o_wp;
  o.BP = o_bp;
  o.K = o_k;

  n.WP = n_wp;
  n.BP = n_bp;
  n.K = n_k;

  MoveListe liste;
  get_moves(o, liste);

  for (int i = 0; i < liste.length(); ++i) {
    Position copy = o;
    copy.make_move(liste[i]);
    if (copy == n) {
      return i;
    }
  }
  // at this point no move was found

  return -1;
}

extern "C" int get_num_moves(uint32_t wp, uint32_t bp, uint32_t k) {
  Position o = Position{};
  Position n = Position{};
  o.WP = wp;
  o.BP = bp;
  o.K = k;

  MoveListe liste;
  get_moves(o, liste);

  return liste.length();
}
