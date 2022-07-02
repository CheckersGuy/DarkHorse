//
// Created by robin on 07.10.21.
//

#ifndef READING_BATCHPROVIDER_H
#define READING_BATCHPROVIDER_H

#include <cstddef>
#include <memory>
#include <PosStreamer.h>
#include <deque>
#include <Weights.h>

//making things a little more readable

class BatchProvider {
private:
    PosStreamer streamer;
    size_t batch_size, buffer_size;
    InputFormat in_format;
public:

    BatchProvider(std::string path, size_t buffer_size, size_t batch_size) : streamer(path, buffer_size,std::make_pair(0,24)) {
        this->batch_size = batch_size;
        this->buffer_size = buffer_size;
    }

    BatchProvider(std::string path, size_t buffer_size, size_t batch_size, size_t a, size_t b) : streamer(path,
                                                                                                          buffer_size,std::make_pair(a,b)) {
        this->batch_size = batch_size;
        this->buffer_size = buffer_size;
    }

    size_t get_batch_size() const;

    size_t get_buffer_size() const;

    std::pair<size_t, size_t> get_piece_range() const;

    PosStreamer &get_streamer();

    void set_input_format(InputFormat format);

    void next(float *results, int64_t *moves, float *inputs);

    void next_pattern(float*results,float* mover,int64_t* op_pawn_index,int64_t* end_pawn_index,int64_t* op_king_index,int64_t* end_king_index,float* wk_input,float*bk_input,float* wp_input,float*bp_input);
};


#endif //READING_BATCHPROVIDER_H
