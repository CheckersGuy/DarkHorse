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
enum class InputFormat{
    V1,V2
};


class BatchProvider {
private:
    PosStreamer streamer;
    size_t batch_size, buffer_size;
public:

    BatchProvider(std::string path, size_t buffer_size, size_t batch_size) : streamer(path, buffer_size, batch_size,std::make_pair(0,24),
                                                                                      true) {
        this->batch_size = batch_size;
        this->buffer_size = buffer_size;
    }

    BatchProvider(std::string path, size_t buffer_size, size_t batch_size, size_t a, size_t b) : streamer(path,
                                                                                                          buffer_size,
                                                                                                          batch_size,std::make_pair(a,b),
                                                                                                          true) {
        this->batch_size = batch_size;
        this->buffer_size = buffer_size;
    }

    size_t get_batch_size() const;

    size_t get_buffer_size() const;

    std::pair<size_t, size_t> get_piece_range() const;

    PosStreamer &get_streamer();

};

class NetBatchProvider : public BatchProvider {

public:
    using BatchProvider::BatchProvider;
    public:
    void next(float *results, int64_t *moves, float *inputs);
};

class NetBatchProvider2 : public BatchProvider{
    using BatchProvider::BatchProvider;
    public:
        void next(float *results, int64_t *moves, float *inputs);
};

#endif //READING_BATCHPROVIDER_H
