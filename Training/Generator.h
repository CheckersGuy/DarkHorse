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
#include <stdlib.h>
#include <unistd.h>
#include <iterator>
#include "Sample.h"



class Generator {
private:
    std::hash<Sample> hash;
    //enough for roughly 200 million insertions
    static constexpr size_t num_bits =5751035027ull;
    static constexpr size_t num_hashes =10;

    size_t buffer_clear_count{100000};
    size_t pos_counter;
    std::string output;
    size_t parallelism{1};
    std::vector<Position> openings;
    int time_control{100};
    int hash_size{21};

    void print_stats();


public:
    Generator(std::string opening, std::string output)
            : output(output) {
        std::string opening_path{"../Training/Positions/"};
        opening_path += opening;

        std::ifstream stream(opening_path,std::ios::binary);
        std::istream_iterator<Position>begin(stream);
        std::istream_iterator<Position>end;
        std::copy(begin,end,std::back_inserter(openings));

        std::cout << "Size: " << openings.size() << std::endl;
    }

    void startx();

    void set_buffer_clear_count(size_t count);

    void set_time(int time);

    void set_parallelism(size_t threads);

    void set_hash_size(int size);

};

#endif //READING_GENERATOR_H
