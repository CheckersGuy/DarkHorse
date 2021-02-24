//
// Created by Robin on 13.06.2017.
//


#include "Transposition.h"

Transposition TT;

Transposition::Transposition(size_t capacity) {
    entries = std::make_unique<Cluster[]>(1u << capacity);
    this->capacity = 1u << (capacity);
    std::fill(entries.get(), entries.get() + capacity, Cluster{});
}

void Transposition::resize(size_t capa) {
    this->entries = std::make_unique<Cluster[]>(1u << capa);
    this->capacity = 1u << capa;
    std::fill(entries.get(), entries.get() + capacity, Cluster{});
}

uint32_t Transposition::getLength() {
    return length;
}

uint32_t Transposition::getCapacity() {
    return capacity;
}

uint32_t Transposition::getHashHits() {
    return hashHit;
}

void Transposition::clear() {
    hashHit = 0;
    std::fill(entries.get(), entries.get() + capacity, Cluster{});
}


double Transposition::getFillRate() {
    return ((double) length) / capacity;
}


void Transposition::storeHash(Value value, uint64_t key, Flag flag, uint8_t depth, uint32_t move_index) {
    this->length++;
    const uint32_t index = (key) & (this->capacity - 1);
    Cluster &cluster = this->entries[index];
    const uint32_t lock = (key >> 32u);


    for (auto i = 1; i < bucket_size; ++i) {
        if (cluster[i].key == lock) {
            cluster[i].age = age_counter;
        }
        const auto age_entry = age_counter - cluster[i].age;

        if (cluster[i].depth < depth || (age_entry != 0)) {
            cluster[i].depth = depth;
            cluster[i].flag = flag;
            cluster[i].bestMove = move_index;
            cluster[i].value = value;
            cluster[i].key = lock;
            cluster[i].age = age_counter;
            return;
        }
    }
    cluster[0].depth = depth;
    cluster[0].flag = flag;
    cluster[0].bestMove = move_index;
    cluster[0].value = value;
    cluster[0].key = lock;
    cluster[0].age = age_counter;
}

bool Transposition::findHash(uint64_t key, NodeInfo &info) {
    const uint32_t index = key & (this->capacity - 1);
    uint32_t currKey = key >> 32u;
    for (int i = 0; i < bucket_size; ++i) {
        if (this->entries[index][i].key == currKey) {
            info.move_index = this->entries[index][i].bestMove;
            info.depth = this->entries[index][i].depth;
            info.flag = this->entries[index][i].flag;
            info.score = this->entries[index][i].value;
            return true;
        }
    }
    info.move_index = Move_Index_None;

    return false;
}

