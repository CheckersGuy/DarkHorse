//
// Created by robin on 07.10.21.
//

#ifndef READING_BATCHPROVIDER_H
#define READING_BATCHPROVIDER_H

#include <cstddef>
#include <memory>
#include <PosStreamer.h>
#include <deque>

enum OutputType{

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
    std::deque<std::shared_ptr<Batch>> batches;
    size_t batch_size, buffer_size;
public:

    BatchProvider(std::string path, size_t buffer_size, size_t batch_size) : streamer(path, buffer_size, batch_size,
                                                                                      true) {
        this->batch_size = batch_size;
        this->buffer_size = buffer_size;
        std::cout<<"Number of training samples: "<<streamer.get_file_size()<<std::endl;
    }

    void next(float *results, float *inputs);

    size_t get_batch_size()const;

    size_t get_buffer_size()const;

    const PosStreamer& get_streamer()const;


};


#endif //READING_BATCHPROVIDER_H
