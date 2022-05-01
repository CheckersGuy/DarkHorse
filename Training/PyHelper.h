//
// Created by root on 06.01.21.
//

#ifndef READING_PYHELPER_H
#define READING_PYHELPER_H



#include <cstdint>
#include "Position.h"
extern "C" void get_next_val_batch(float *results,int64_t *moves, float *inputs);
extern "C" void get_next_batch(float *results,int64_t *moves,float *inputs);
extern "C" void get_next_val_batch2(float *results,int64_t *moves, float *inputs);
extern "C" void get_next_batch2(float *results,int64_t *moves,float *inputs);
extern "C" int init_val_streamer(size_t buffer_size, size_t batch_size,size_t a,size_t b, char *file_path,bool patterns);
extern "C" int init_streamer(size_t buffer_size, size_t batch_size,size_t a,size_t b, char *file_path,bool patterns);
extern "C" int init_val_streamer2(size_t buffer_size, size_t batch_size,size_t a,size_t b, char *file_path,bool patterns);
extern "C" int init_streamer2(size_t buffer_size, size_t batch_size,size_t a,size_t b, char *file_path,bool patterns);

extern "C" void print_fen(const char* fen_string);
extern "C" void get_input_from_fen(float* inputs,const char* fen_string);


#endif //READING_PYHELPER_H
