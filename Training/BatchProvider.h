//
// Created by robin on 07.10.21.
//

#ifndef READING_BATCHPROVIDER_H
#define READING_BATCHPROVIDER_H

#include "types.h"
#include <cstddef>
#include <memory>
#include <PosStreamer.h>
#include <deque>


//making things a little more readable

class BatchProvider {
private:
    PosStreamer streamer;
    size_t batch_size, buffer_size;
public:

    BatchProvider(std::string path, size_t buffer_size, size_t batch_size):streamer(path,buffer_size,getSystemTime()) {
        this->batch_size = batch_size;
        this->buffer_size = buffer_size;
        Zobrist::init_zobrist_keys(getSystemTime());
    }

    BatchProvider(std::string path, size_t buffer_size, size_t batch_size, size_t a, size_t b) : streamer(path,
                                                                                                          buffer_size,getSystemTime()) {
        this->batch_size = batch_size;
        this->buffer_size = buffer_size;
        Zobrist::init_zobrist_keys(getSystemTime());
    }

    size_t get_batch_size() const;

    size_t get_buffer_size() const;

    PosStreamer &get_streamer();

    void next(float *results, int64_t *moves, float *inputs);

};


#endif //READING_BATCHPROVIDER_H
