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
#include <chrono>



class PosStreamer {

private:
    size_t gen_seed;
    std::string file_path;
    size_t buffer_size;
    std::vector<Sample> buffer;
    size_t ptr;
    std::ifstream stream;
    std::mt19937_64 generator;
    bool shuffle{true};
    size_t num_samples; // number of samples
    std::vector<Game> games;
    size_t game_offset{0};
    size_t random_skip{5};

public:

    PosStreamer(std::string file_path, size_t buff_size, size_t seed = 12312312){
                
        this->file_path = file_path;
        gen_seed=seed;
        ptr = buffer_size + 1;
        stream = std::ifstream(file_path, std::ios::binary);
        generator = std::mt19937_64(getSystemTime());
        if (file_path.empty()) {
            std::cerr << "An empty path was given" << std::endl;
            std::exit(-1);
        }
        if (!stream.good()) {
            std::cerr << "Could not open the stream" << std::endl;
            std::cerr << "FileName: " << file_path << std::endl;
            std::exit(-1);
        }
        //loading the game
        std::istream_iterator<Game> begin(stream);
        std::istream_iterator<Game>end;
        std::cout<<"Loading games"<<std::endl;
        std::copy(begin,end,std::back_inserter(games));
        std::cout<<"Done loading games"<<std::endl;
        std::shuffle(games.begin(),games.end(),generator);
        std::cout<<"Done shuffling"<<std::endl;
        num_samples = count_trainable_positions(file_path);
                std::cout<<"loading"<<std::endl;
        this->buffer_size = std::min(num_samples, buff_size);
        std::cout<<"BufferSize: "<<buffer_size<<std::endl;
        buffer.reserve(buffer_size);
        std::cout<<buffer_size<<std::endl;

    }

    Sample get_next();

    void set_shuffle(bool shuff);

    size_t get_buffer_size() const;

    size_t ptr_position();

    size_t get_file_size() const;

    size_t get_num_positions() const;

    const std::string &get_file_path();



};

#endif //READING_POSSTREAMER_H
