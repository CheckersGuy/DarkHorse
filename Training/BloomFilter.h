//
// Created by robin on 31.08.21.
//

#ifndef READING_BLOOMFILTER_H
#define READING_BLOOMFILTER_H

#include <vector>
#include "Sample.h"

template<typename T, typename Hash = std::hash<T>>
class BloomFilter {

private:
    Hash hash;
    std::vector<bool> bit_set;
    size_t num_bits;
    size_t num_hashes;
    size_t num_insertions{0};
public:

    BloomFilter(size_t num_bits, size_t num_hashes) : num_bits(num_bits), num_hashes(num_hashes),
                                                      bit_set(num_bits, false) {
    }


    void clear(){
        for(auto i=size_t{0};i<bit_set.size();++i){
            bit_set[i]=false;
        }
        num_insertions=0;
    }

    size_t get_num_insertions() {
        return num_insertions;
    }


    size_t get_num_bits() {
        return num_bits;
    }

    size_t get_num_hashes() {
        return num_hashes;
    }

    BloomFilter(const BloomFilter &filter) {
        num_bits = filter.num_bits;
        num_hashes = filter.num_hashes;
        std::copy(filter.bit_set.begin(), filter.bit_set.end(), std::back_inserter(bit_set));
        num_insertions = filter.insertions;
    }

    BloomFilter &operator=(BloomFilter &filter) {
        num_bits = filter.num_bits;
        num_hashes = filter.num_hashes;
        num_insertions = filter.insertions;
        std::copy(filter.bit_set.begin(), filter.bit_set.end(), std::back_inserter(bit_set));
        return *this;
    }

    void insert(T sample) {
        num_insertions++;
        uint64_t hash_val = hash(sample);
        auto hash1 = static_cast<uint32_t>(hash_val);
        auto hash2 = static_cast<uint32_t>(hash_val >> 32);

        for (uint32_t k = 0; k < num_hashes; ++k) {
            uint32_t current_hash = hash1 + hash2 * k;
            size_t index = current_hash % num_bits;
            bit_set[index] = true;
        }

    }

    bool has(const T &other) {
        uint64_t hash_val = hash(other);
        auto hash1 = static_cast<uint32_t>(hash_val);
        auto hash2 = static_cast<uint32_t>(hash_val >> 32);
        for (uint32_t k = 0; k < num_hashes; ++k) {
            uint32_t current_hash = hash1 + hash2 * k;
            size_t index = current_hash % num_bits;
            if (bit_set[index] == false)
                return false;
        }
        return true;
    }

};


#endif //READING_BLOOMFILTER_H
