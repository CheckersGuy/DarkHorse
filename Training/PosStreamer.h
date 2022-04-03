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

class PosStreamer {

private:
    std::string file_path;
    const size_t buffer_size;
    std::unique_ptr<Sample[]> buffer;
    size_t ptr;
    std::ifstream stream;
    std::mt19937_64 generator;
    const bool shuffle;
    size_t file_size; // number of samples

private:
public:

    PosStreamer(std::string file_path, size_t buffer_size, bool shuffle = true, size_t seed = 231231231ull)
            : buffer_size(buffer_size), shuffle(shuffle),file_path(file_path) {

        //computing the file_size
        ptr = buffer_size + 1;
        file_size = PosStreamer::get_file_size() / sizeof(Sample);
        stream = std::ifstream(file_path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Could not open the stream" << std::endl;
            std::exit(-1);
        }
        buffer = std::make_unique<Sample[]>(buffer_size);
        generator = std::mt19937_64(seed);
    }

    Sample get_next();

    size_t get_buffer_size() const;

    size_t ptr_position();

    size_t get_file_size() const;

    size_t get_num_positions() const;

    const std::string &get_file_path();

};

#endif //READING_POSSTREAMER_H
