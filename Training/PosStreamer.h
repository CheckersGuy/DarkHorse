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
class PosStreamer {

private:
    std::string file_path;
    const size_t buffer_size;
    std::vector<Sample> buffer;
    size_t ptr;
    std::ifstream stream;
    std::mt19937_64 generator;
    const bool shuffle;
    size_t num_samples; // number of samples

private:
public:

    PosStreamer(std::string file_path, size_t buffer_size, bool shuffle = true, size_t seed = 231231231ull)
            : buffer_size(buffer_size), shuffle(shuffle),file_path(file_path) {

        //computing the file_size
        ptr = buffer_size + 1;
        stream = std::ifstream(file_path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Could not open the stream" << std::endl;
            std::exit(-1);
        }
        generator = std::mt19937_64(seed);
        num_samples = count_trainable_positions(file_path);
    }

    Sample get_next();

    size_t get_buffer_size() const;

    size_t ptr_position();

    size_t get_file_size() const;

    size_t get_num_positions() const;

    const std::string &get_file_path();

};

#endif //READING_POSSTREAMER_H
