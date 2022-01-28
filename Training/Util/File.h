//
// Created by robin on 18.12.21.
//

#ifndef READING_FILE_H
#define READING_FILE_H

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include "fcntl.h"
#include <string>
#include <fstream>
#include <iterator>
#include <Sample.h>
#include <BloomFilter.h>
#include <string>
#include <memory>
#include <filesystem>

namespace File {
    void remove_duplicates(std::string input, std::string output);

    size_t num_illegal_samples(std::string input);

    template<typename T>
    void merge_files(std::initializer_list<std::string> files, std::string output) {
        std::ofstream out_stream(output, std::ios::binary);

        for (auto &file: files) {
            std::ifstream stream(file, std::ios::binary);
            std::istream_iterator<T> begin(stream);
            std::istream_iterator<T> end;
            std::copy(begin, end, std::ostream_iterator<T>(out_stream));
        }

    }


    template<typename T>
    void external_shuffle(std::string input) {
        struct stat stat_buf;
        int rc = stat(input.c_str(), &stat_buf);
        auto num_elements = rc / (sizeof(T));
        auto fd = open(input.c_str(), O_RDWR);

        std::filesystem::path my_path(input);
        auto size = std::filesystem::file_size(my_path);
        auto num_examples = size / (sizeof(T));

        T *elements = (Sample *) mmap(NULL, sizeof(Sample) * num_examples, PROT_READ | PROT_WRITE, MAP_SHARED, fd,
                                      0);


        //doing the shiffer yates shuffle
        std::mt19937_64 generator(11123123ull);
        auto t1 = std::chrono::duration_cast<std::chrono::milliseconds>
                (std::chrono::high_resolution_clock::now().time_since_epoch()).count();
        for (auto i = num_examples - 1; i >= 1; i--) {
            //getting a random number between 0 and i

            auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>
                    (std::chrono::high_resolution_clock::now().time_since_epoch()).count();;
            auto dur = (t2 - t1);
            if (dur >= 1000) {
                t1 = t2;

                std::cout << "Counter: " << i<< "\n";
            }

            std::uniform_int_distribution<size_t> distrib(0, i);
            auto j = distrib(generator);
            auto temp = elements[i];
            elements[i] = elements[j];
            elements[j] = temp;
        }


    }
}
#endif //READING_FILE_H
