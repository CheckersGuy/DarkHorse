//
// Created by robin on 07.10.21.
//

#ifndef READING_BATCHPROVIDER_H
#define READING_BATCHPROVIDER_H

#include <cstddef>
#include <memory>
#include <PosStreamer.h>
#include <deque>

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
    std::deque<std::shared_ptr<Batch>> batches;
    size_t batch_size;
public:

    void next(float *results, float *inputs);


};


#endif //READING_BATCHPROVIDER_H
