//
// Created by robin on 02.09.21.
//

#ifndef READING_SAMPLEFILTER2_H
#define READING_SAMPLEFILTER2_H
#include <vector>
#include <Generator.h>


class SampleFilter2 {


private:
    std::hash<Sample> hash;
    std::unique_ptr<uint8_t[]>bit_set;
    size_t num_bits;
    size_t num_hashes;

    void set(size_t index);

    bool get(size_t index);


public:

    SampleFilter2(size_t num_bits, size_t num_hashes) : num_bits(num_bits), num_hashes(num_hashes),bit_set(std::make_unique<uint8_t[]>(num_bits/8 +1)) {
    }

    SampleFilter2() = default;

    SampleFilter2(const SampleFilter2 &filter);

    size_t get_num_bits();

    size_t get_num_hashes();

    void insert(Sample sample);

    SampleFilter2 &operator=(SampleFilter2 &other);

    bool has(const Sample &other);

};


#endif //READING_SAMPLEFILTER2_H
