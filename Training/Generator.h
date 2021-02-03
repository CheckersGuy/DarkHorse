//
// Created by root on 03.02.21.
//

#ifndef READING_GENERATOR_H
#define READING_GENERATOR_H
#include <iostream>
#include <string>
#include "Position.h"
#include "Board.h"
#include "GameLogic.h"
class Generator {
private:
    std::string engine_path;
    std::string opening_book;
    std::string output;
    size_t num_games;
    size_t parallelism{1};
    std::vector<std::pair<Position,int>> buffer;
    //Position and Result
    using Sample = std::pair<Position,int>;
    int avg_length{0};
public:

    Generator(std::string engine, std::string opening,std::string output="test_data.dat"){

    }

    static std::pair<std::vector<Position>,int> generate_game(Position start_pos,int time_c);

    void start();

};


#endif //READING_GENERATOR_H
