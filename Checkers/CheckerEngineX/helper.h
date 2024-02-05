#include "Endgame.h"
#include "Position.h"
#include "types.h"
inline TableBase base;

extern "C" void load(char *path, int cache_size, int num_pieces);

extern "C" void load_dtw(char *dtw_path, int cache_size,
                         int num_pieces); // loads both wdl and dtw if available
extern "C" void close();
extern "C" void print_fen(char *fen_string);
extern "C" int probe(char *fen_string);
extern "C" int probe_dtw(char *fen_string);
