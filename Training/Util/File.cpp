//
// Created by robin on 18.12.21.
//


#include "File.h"

namespace File {


    bool file_equal(std::string file_one, std::string file_two) {
        std::ifstream stream_one(file_one, std::ios::binary);
        std::ifstream stream_two(file_two, std::ios::binary);

        std::istreambuf_iterator<char> begin_one(stream_one);
        std::istreambuf_iterator<char> begin_two(stream_two);

        return std::equal(begin_one, std::istreambuf_iterator<char>{}, begin_two);
    }

}