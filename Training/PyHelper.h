//
// Created by root on 06.01.21.
//

#ifndef READING_PYHELPER_H
#define READING_PYHELPER_H
#include "Board.h"
#include <cstdint>
#include "Position.h"
extern "C" void get_next_val_batch(float *results,int64_t *moves, float *inputs);
extern "C" void get_next_batch(float *results,int64_t *moves,float *inputs);


extern "C" int init_streamer(size_t buffer_size, size_t batch_size, char *file_path);
extern "C" int init_val_streamer(size_t buffer_size, size_t batch_size, char *file_path);
extern "C" void print_fen(const char* fen_string);
extern "C" void get_input_from_fen(float* inputs,const char* fen_string);

extern "C" int get_bucket_index(uint32_t bp,uint32_t wp,uint32_t k);

extern "C" Board * create_board();

extern "C" void delete_board(Board* board);

extern "C" void print_board(Board* board);


#endif //READING_PYHELPER_H
