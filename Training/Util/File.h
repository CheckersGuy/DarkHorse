//
// Created by robin on 18.12.21.
//

#ifndef READING_FILE_H
#define READING_FILE_H
#include <string>
#include <fstream>
#include <iterator>
#include <Sample.h>
#include <SampleFilter.h>
void remove_duplicates(std::string input, std::string output);

template<typename T>
void merge_files(std::initializer_list<std::string> files, std::string output) {
    std::ofstream out_stream(output,std::ios::binary);

    for(auto& file : files){
        std::ifstream stream(file,std::ios::binary);
        std::istream_iterator<T> begin(stream);
        std::istream_iterator<T>end;
        std::copy(begin,end,std::ostream_iterator<T>(out_stream));
    }

}

#endif //READING_FILE_H
