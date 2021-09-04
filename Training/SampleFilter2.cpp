//
// Created by robin on 02.09.21.
//

#include "SampleFilter2.h"

size_t SampleFilter2::get_num_hashes() {
    return num_hashes;
}

size_t SampleFilter2::get_num_bits() {
    return num_bits;
}

void SampleFilter2::set(size_t index) {
    const size_t part_index = index / 8u;
    const size_t sub_index = index % 8u;
    const uint8_t maske = (1u << sub_index) | bit_set[part_index];
    bit_set[part_index] = maske;
}

bool SampleFilter2::get(size_t index) {
    const size_t part_index = index / 8u;
    const size_t sub_index = index % 8u;
    const uint8_t maske = 1u << sub_index;
    return ((bit_set[part_index] & maske) == maske);
}

void SampleFilter2::insert(Sample sample) {
    uint64_t hash_val = hash(sample);

    auto hash1 = static_cast<uint32_t>(hash_val);
    auto hash2 = static_cast<uint32_t>(hash_val >> 32);

    for (uint32_t k = 0; k < num_hashes; ++k) {
        uint32_t current_hash = hash1 + hash2 * k;
        size_t index = current_hash % num_bits;
        set(index);
    }
}

bool SampleFilter2::has(const Sample &other) {
    uint64_t hash_val = hash(other);
    //extracing lower and upper 32 bits
    auto hash1 = static_cast<uint32_t>(hash_val);
    auto hash2 = static_cast<uint32_t>(hash_val >> 32);
    for (uint32_t k = 0; k < num_hashes; ++k) {
        uint32_t current_hash = hash1 + hash2 * k;
        size_t index = current_hash % num_bits;
        if (get(index) == false)
            return false;
    }
    return true;
}