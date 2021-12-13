//
// Created by root on 06.01.21.
//

#ifndef READING_PYHELPER_H
#define READING_PYHELPER_H



#include <cstdint>
#include "Position.h"
extern "C" void get_next_val_batch(float *results,int64_t *moves, float *inputs);
extern "C" void get_next_batch(float *results,int64_t *moves,float *inputs);
extern "C" int init_val_streamer(size_t buffer_size, size_t batch_size, char *file_path,bool patterns);
extern "C" int init_streamer(size_t buffer_size, size_t batch_size, char *file_path,bool patterns);


extern "C" int get_bucket_index(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" void print_fen(int color, uint32_t white_men, uint32_t black_men, uint32_t kings);



extern "C" void print_position(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int count_bits(uint32_t bitfield);
extern "C" int count_white_kings(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int count_black_kings(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int count_white_pawn(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int count_black_pawn(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int invert_pieces(uint32_t pieces);
extern "C" int has_jumps(uint32_t white_men, uint32_t black_men, uint32_t kings, int color);


#endif //READING_PYHELPER_H
