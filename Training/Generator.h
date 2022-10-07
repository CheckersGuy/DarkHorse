//
// Created by root on 03.02.21.
//

#ifndef READING_GENERATOR_H
#define READING_GENERATOR_H

#include <unistd.h>
#include <fstream>
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
#include <cstdlib>
#include <unistd.h>
#include <iterator>
#include "Sample.h"
#include <Util/Compress.h>

class Generator {
private:
    size_t buffer_clear_count{100000};
    size_t max_positions{0};
    size_t piece_lim{0};
    size_t max_games{1000000};
    std::string output;
    size_t parallelism{1};
    std::vector<Position> openings;
    int time_control{100};
    int hash_size{21};
    uint64_t *random;
    pthread_mutex_t *pmutex;


    uint64_t get_shared_random_number();

public:
    Generator(std::string opening, std::string output)
            : output(output) {
        std::string opening_path{"../Training/Positions/"};
        opening_path += opening;

        std::ifstream stream(opening_path, std::ios::binary);
        std::istream_iterator<Position> begin(stream);
        std::istream_iterator<Position> end;
        std::copy(begin, end, std::back_inserter(openings));
        std::cout << "Size: " << openings.size() << std::endl;
    }

    Generator()=default;

    void start();

    void set_book(std::string book);

    void set_output(std::string output);

    void set_buffer_clear_count(size_t count);

    void set_time(int time);

    void set_parallelism(size_t threads);

    void set_hash_size(int size);

    void set_max_position(size_t max);

    void set_max_games(size_t max_games);

    void set_piece_limit(size_t num_pieces);

};

#endif //READING_GENERATOR_H
