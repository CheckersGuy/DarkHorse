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
        do {
            if (game_offset>=games.size()) {
                game_offset =0;
                std::shuffle(games.begin(),games.end(),generator);
            }
            auto game = games[game_offset++];
            //adding filter so we can duplicate endgame positions
            game.extract_samples(std::back_inserter(buffer));
        } while (buffer.size() < buffer_size);
        if (shuffle) {
            auto t1 = std::chrono::high_resolution_clock::now();
            std::shuffle(buffer.begin(), buffer.end(), generator);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto dur = t2-t1;
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

void PosStreamer::set_shuffle(bool shuff) {
    shuffle=shuff;
}

