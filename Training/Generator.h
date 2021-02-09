//
// Created by root on 03.02.21.
//

#ifndef READING_GENERATOR_H
#define READING_GENERATOR_H

#include <unistd.h>
#include "Utilities.h"
#include <iostream>
#include <string>
#include "Position.h"
#include "Board.h"
#include "GameLogic.h"
#include  <sys/types.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <types.h>

struct Sample {
    Position position;
    int result{1000};

    friend std::ostream &operator<<(std::ostream &stream, const Sample &s);

    friend std::istream &operator>>(std::istream &stream, Sample &s);

};

struct Instance {

    enum State {
        Idle, Generating, Error, Init
    };


    const int &read_pipe;
    const int &write_pipe;
    bool waiting_response = false;
    State state{Idle};

    void write_message(std::string msg);

    std::string read_message();

};


class Generator {
private:
    static constexpr size_t BUFFER_SIZE = 50000;
    std::string engine_path;
    std::string output;
    size_t num_games{100000};
    size_t game_counter{0},num_wins{0};
    size_t parallelism{1};
    std::vector<Instance> instances;
    std::vector<Sample> buffer;
    std::vector<Position> openings;
    size_t opening_index{0};
    int time_control{100};
public:
    Generator(std::string engine, std::string opening, std::string output)
            : engine_path(engine), output(output) {
        std::string opening_path{"../Training/Positions/"};
        opening_path += opening;
        Utilities::read_binary<Position>(std::back_inserter(openings), opening_path);
        std::cout << "Size: " << openings.size() << std::endl;
    }

    void start();

    void process();

    void clearBuffer();

    void set_time(int time);

    void set_parallelism(size_t threads);

    void set_num_games(size_t num_games);

    void print_stats();

    Position get_start_pos();


};

#endif //READING_GENERATOR_H
