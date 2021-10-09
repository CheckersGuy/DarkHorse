//
// Created by robin on 31.08.21.
//

#ifndef READING_SAMPLEFILTER_H
#define READING_SAMPLEFILTER_H

#include <vector>
#include "Sample.h"
class SampleFilter {

private:
    std::hash<Sample> hash;
    std::vector<bool> bit_set;
    size_t num_bits;
    size_t num_hashes;
    size_t num_insertions{0};
    size_t unique_count{0};

public:

    SampleFilter(size_t num_bits, size_t num_hashes) : num_bits(num_bits), num_hashes(num_hashes),
                                                       bit_set(num_bits, false) {
    }

    SampleFilter(const SampleFilter &filter);


    size_t get_num_bits();

    size_t get_num_hashes();

    size_t get_num_insertions();

    void insert(Sample sample);

    SampleFilter &operator=(SampleFilter &other);

    bool has(const Sample &other);

};


#endif //READING_SAMPLEFILTER_H
