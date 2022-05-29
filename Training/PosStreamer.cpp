//
// Created by robin on 06.10.21.
//

#include <sys/stat.h>
#include "PosStreamer.h"

size_t PosStreamer::get_num_positions() const {
    return num_samples;
}

void PosStreamer::set_input_format(InputFormat format){
    in_format = format;
}

InputFormat PosStreamer::get_input_format()const{
    return in_format;
}

Sample PosStreamer::get_next() {
    if (ptr >= buffer_size) {
        buffer.clear();
        ptr = 0;
        do {
            if (stream.peek() == EOF) {
                stream.clear();
                stream.seekg(0, std::ios::beg);
            }
            Game game;
            stream >> game;
            game_buffer.clear();
            auto lambda =[&](Color color, Move move)->int{
                if(in_format == InputFormat::V1){
                    return Statistics::MovePicker::get_move_encoding(color,move);
                }else if(in_format == InputFormat::V2){
                    //simple policy encoding of all possible from-to moves

                    if(color == BLACK){
                        move.from = getMirrored(move.from);
                        move.to = getMirrored(move.to);
                        move.captures = getMirrored(move.captures);
                        return 32*move.get_from_index() + move.get_to_index();
                    }else{
                        return 32*move.get_from_index() + move.get_to_index();
                    }
                }
            };

            game.extract_samples_test(std::back_inserter(game_buffer),lambda);
            for (auto &s: game_buffer) {
                auto num_p = Bits::pop_count(s.position.BP | s.position.WP);
                if (num_p > range.second) {
                    continue;
                }
                if (num_p < range.first) {
                    break;
                }
                buffer.emplace_back(s);
            }
        } while (buffer.size() < buffer_size);
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

std::pair<size_t, size_t> PosStreamer::get_range() const {
    return range;
}

void PosStreamer::set_shuffle(bool shuff){
    shuffle=shuff;
}

