
#ifdef _WIN32
#include "Endgame.h"
#endif

#include "Position.h"
#include "types.h"

#ifdef _WIN32
inline TableBase base;

extern "C" void load(char *path, int cache_size, int num_pieces);

extern "C" void load_dtw(char *dtw_path, int cache_size, int num_pieces);

extern "C" void close();
extern "C" int probe(char *fen_string);
extern "C" int probe_dtw(char *fen_string);
extern "C" int probe_with_position(unsigned int bp, unsigned int wp,
                                   unsigned int k, int color);
extern "C" int probe_dtw_with_position(unsigned int bp, unsigned int wp,
                                       unsigned int k, int color);

#endif
extern "C" void print_fen(char *fen_string);
extern "C" int move_played(char *orig, char *next);

extern "C" int move_played_pos(uint32_t o_wp, uint32_t o_bp, uint32_t o_k,
                               uint32_t n_wp, uint32_t n_bp, uint32_t n_k);

extern "C" int get_num_moves(uint32_t wp, uint32_t bp, uint32_t k);
