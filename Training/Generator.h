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
#include  <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>

class Generator {
private:
    static constexpr size_t BUFFER_SIZE = 5000;
    std::string engine_path;
    std::string opening_book;
    std::string output;
    size_t num_games{100000};
    size_t parallelism{1};
    std::vector<std::pair<Position, int>> buffer;
    //Position and Result
    int avg_length{0};
public:
    using Sample = std::pair<Position, int>;

    Generator(std::string engine, std::string opening, std::string output)
            : engine_path(engine), opening_book(opening), output(output) {}

    void start();

    void clearBuffer();

    void set_time(int time);

    void set_parallelism(size_t threads);

    void set_num_games(size_t num_games);

};


#endif //READING_GENERATOR_H
