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
#include <sys/mman.h>
#include <stdlib.h>
#include <unistd.h>

struct Sample {
    Position position;
    int result{1000};

    friend std::ostream &operator<<(std::ostream &stream, const Sample &s);

    friend std::istream &operator>>(std::istream &stream, Sample &s);

    bool operator==(const Sample &other) const;

    bool operator!=(const Sample &other) const;
};

struct SampleHasher {
    std::hash<int> hasher;

    uint64_t operator()(Sample s) const {
        return Zobrist::generateKey(s.position) ^ hasher(s.result);
    }
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
    size_t parallelism{1};
    size_t max_games;
    std::vector<Instance> instances;
    std::vector<Sample> buffer;
    std::vector<Position> openings;
    size_t opening_index{0};
    int time_control{100};
    int hash_size{21};
public:
    Generator(std::string engine, std::string opening, std::string output)
            : engine_path(engine), output(output) {
        std::string opening_path{"../Training/Positions/"};
        opening_path += opening;
        Utilities::read_binary<Position>(std::back_inserter(openings), opening_path);
        std::cout << "Size: " << openings.size() << std::endl;
    }

    void start();

    void startx();

    void set_time(int time);

    void set_parallelism(size_t threads);

    void set_num_games(size_t num_games);

    void set_hash_size(int size);

};

#endif //READING_GENERATOR_H
