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
enum class InputFormat : int{
    //V1 is for the nnue nets
    //V2 is for convnets
    //V3 is for pattern based eval
    V1=0,V2=1,PATTERN =2
};




class PosStreamer {

private:
    InputFormat in_format{InputFormat::PATTERN};
    size_t gen_seed;
    std::string file_path;
    size_t buffer_size;
    std::vector<Sample> buffer;
    size_t ptr;
    std::ifstream stream;
    std::mt19937_64 generator;
    bool shuffle{true};
    size_t num_samples; // number of samples
    std::pair<size_t, size_t> range;
    std::vector<Game> games;
    size_t game_offset{0};

public:

    PosStreamer(std::string file_path, size_t buff_size,
                std::pair<size_t, size_t> ran = std::make_pair(0, 24), size_t seed = 12312312){
                
        this->file_path = file_path;
        gen_seed=seed;
        range = ran;
        ptr = buffer_size + 1;
        stream = std::ifstream(file_path, std::ios::binary);
        generator = std::mt19937_64(seed);
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
        std::cout<<"Memory for games: "<<(games.capacity()*sizeof(Game)/1000000)<<std::endl;
        num_samples = count_trainable_positions(file_path, range);
                std::cout<<"loading"<<std::endl;
        this->buffer_size = std::min(num_samples, buff_size);
        std::cout<<"BufferSize: "<<buffer_size<<std::endl;
        std::cout<<"Range of positions"<<"["<<range.first<<","<<range.second<<"]"<<std::endl;
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

    std::pair<size_t, size_t> get_range() const;

    void set_input_format(InputFormat format);

    InputFormat get_input_format()const;


};

#endif //READING_POSSTREAMER_H
