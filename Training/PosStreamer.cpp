//
// Created by robin on 06.10.21.
//

#include <sys/stat.h>
#include "PosStreamer.h"

size_t PosStreamer::get_num_positions() const {
    return num_samples;
}

Sample PosStreamer::get_next() {
    if (ptr >= buffer_size) {
        buffer.clear();
        ptr = 0;
        //if we reached the end of our file
        //we have to wrap around
        size_t read_elements = 0;
        do {
            if (stream.peek() == EOF) {
                stream.clear();
                stream.seekg(0, std::ios::beg);
            }
            Game game;
            stream >> game;
            std::vector<Sample> game_samples;
            game.extract_samples(std::back_inserter(game_samples));
            for (auto s: game_samples) {
                buffer.emplace_back(s);
            }
        } while (buffer.size() < buffer_size);
        //the buffer is filled now so we can shuffle the elements
        if (shuffle) {
            std::shuffle(buffer.begin(), buffer.end(), generator);
        }
    }
    return buffer[ptr++];
}


size_t PosStreamer::get_buffer_size() const {
    return buffer_size;
}

size_t PosStreamer::ptr_position() {
    return ptr;
}

const std::string &PosStreamer::get_file_path() {
    return file_path;
}


size_t PosStreamer::get_file_size() const {
    std::filesystem::path my_path(file_path);
    return std::filesystem::file_size(my_path);
}


