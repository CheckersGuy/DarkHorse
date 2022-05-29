//
// Created by robin on 06.10.21.
//

#ifndef READING_POSSTREAMER_H
#define READING_POSSTREAMER_H

#include <string>
#include <memory>
#include <../CheckerEngineX/Position.h>
#include <fstream>
#include <Sample.h>
#include <iterator>
#include <filesystem>
#include "Util/Compress.h"

//the only difference between these two formats is the policy encoding of the moves
//the nnue-net is not trained on captures and hence the range of possible moves is much less.
//A conv-policy net will have an output of size 32*31 (from-to square encoding) 
enum class InputFormat : int{
    V1=0,V2=1
};




class PosStreamer {

private:
    InputFormat in_format{InputFormat::V1};
    std::string file_path;
    size_t buffer_size;
    std::vector<Sample> buffer;
    size_t ptr;
    std::ifstream stream;
    std::mt19937_64 generator;
    bool shuffle;
    size_t num_samples; // number of samples
    std::pair<size_t, size_t> range;
    std::vector<Sample> game_buffer;

public:

    PosStreamer(std::string file_path, size_t buffer_size, bool shuffle = true,
                std::pair<size_t, size_t> ran = std::make_pair(0, 24), size_t seed = 431231231ull)
            : buffer_size(buffer_size), shuffle(shuffle), file_path(file_path) {

        //computing the file_size
        range = ran;
        ptr = buffer_size + 1;
        stream = std::ifstream(file_path, std::ios::binary);
        if (file_path.empty()) {
            std::cerr << "An empty path was given" << std::endl;
            std::exit(-1);
        }
        if (!stream.good()) {
            std::cerr << "Could not open the stream" << std::endl;
            std::cerr << "FileName: " << file_path << std::endl;
            std::exit(-1);
        }
        generator = std::mt19937_64(seed);
        num_samples = count_trainable_positions(file_path, range);
        buffer_size = std::min(num_samples, buffer_size);
        buffer.reserve(buffer_size);
        game_buffer.reserve(200);
    }

    Sample get_next();

    void set_shuffle(bool shuff);

    size_t get_buffer_size() const;

    size_t ptr_position();

    size_t get_file_size() const;

    size_t get_num_positions() const;

    const std::string &get_file_path();

    std::pair<size_t, size_t> get_range() const;

    void set_input_format(InputFormat format);

    InputFormat get_input_format()const;


};

#endif //READING_POSSTREAMER_H
