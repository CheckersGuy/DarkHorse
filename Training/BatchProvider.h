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

enum OutputType {

};

struct Batch {
    const size_t batch_size;
    static constexpr size_t input_size = 120;
    static constexpr size_t result_size = 1;
    std::unique_ptr<float[]> results;
    std::unique_ptr<float[]> inputs;
    //all the other stuff

    Batch(size_t size) : batch_size(size) {
        results = std::make_unique<float[]>(batch_size * result_size);
        inputs = std::make_unique<float[]>(batch_size * input_size);
    }

};


class BatchProvider {
private:
    PosStreamer streamer;
    size_t batch_size, buffer_size;
public:

    BatchProvider(std::string path, size_t buffer_size, size_t batch_size) : streamer(path, buffer_size, batch_size,
                                                                                      true) {
        this->batch_size = batch_size;
        this->buffer_size = buffer_size;
        std::cout << "Number of training samples: " << streamer.get_file_size() << std::endl;
    }

    size_t get_batch_size() const;

    size_t get_buffer_size() const;

    PosStreamer &get_streamer();

};

class NetBatchProvider : public BatchProvider {

public:
    using BatchProvider::BatchProvider;

    void next(float *results,int64_t * moves,float *inputs);
};

class PattBatchProvider : public BatchProvider {
public:

    using BatchProvider::BatchProvider;

    void next(float *results, float *num_wp, float *num_bp, float *num_wk, float *num_bk, int64_t *patt_op_big,
              int64_t *patt_end_big, int64_t *patt_op_small,
              int64_t *patt_end_small);
};




#endif //READING_BATCHPROVIDER_H
