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
#include "Sample.h"



class Generator {
private:
    std::hash<Sample> hash;
    //enough for roughly 200 million insertions
    static constexpr size_t num_bits =5751035027ull;
    static constexpr size_t num_hashes =10;

    size_t buffer_clear_count{100000};
    size_t past_uniq_counter;
    size_t pos_counter;
    uint8_t * bit_set;
    std::string output;
    size_t parallelism{1};
    std::vector<Position> openings;
    int time_control{100};
    int hash_size{21};
    void set(size_t index);

    bool get(size_t index);

    void insert(Sample s);

    bool has(const Sample& s);

    void save_filter(size_t uniq_pos_seen,size_t pos_seen);

    void load_filter();

    void print_stats();


public:
    Generator(std::string opening, std::string output)
            : output(output) {
        std::string opening_path{"../Training/Positions/"};
        opening_path += opening;
        Utilities::read_binary<Position>(std::back_inserter(openings), opening_path);
        std::cout << "Size: " << openings.size() << std::endl;
    }

    void startx();

    void set_buffer_clear_count(size_t count);

    void set_time(int time);

    void set_parallelism(size_t threads);

    void set_hash_size(int size);

    //temporary to switch to the new approach in case we have no previous filter file
    void create_filter_file(std::string input);

};

#endif //READING_GENERATOR_H
