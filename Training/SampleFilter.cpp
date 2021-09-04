//
// Created by robin on 31.08.21.
//

#include "SampleFilter.h"

size_t SampleFilter::get_num_bits() {
    return num_bits;
}

size_t SampleFilter::get_num_hashes() {
    return num_hashes;
}

SampleFilter::SampleFilter(const SampleFilter &filter) {
    num_bits = filter.num_bits;
    num_hashes = filter.num_hashes;
    std::copy(filter.bit_set.begin(), filter.bit_set.end(), std::back_inserter(bit_set));
}

SampleFilter &SampleFilter::operator=(SampleFilter &filter) {
    num_bits = filter.num_bits;
    num_hashes = filter.num_hashes;
    std::copy(filter.bit_set.begin(), filter.bit_set.end(), std::back_inserter(bit_set));
    return *this;
}

void SampleFilter::insert(Sample sample) {
    uint64_t hash_val = hash(sample);
    //extracing lower and upper 32 bits
    auto hash1 = static_cast<uint32_t>(hash_val);
    auto hash2 = static_cast<uint32_t>(hash_val >> 32);

    for (uint32_t k = 0; k < num_hashes; ++k) {
        uint32_t current_hash = hash1 + hash2 * k;
        size_t index = current_hash % num_bits;
        bit_set[index] = true;
    }

}

bool SampleFilter::has(const Sample &other) {
    uint64_t hash_val = hash(other);
    //extracing lower and upper 32 bits
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