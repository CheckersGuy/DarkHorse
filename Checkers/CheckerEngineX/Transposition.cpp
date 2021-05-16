//
// Created by Robin on 13.06.2017.
//


#include "Transposition.h"

Transposition TT;

Transposition::Transposition(size_t capacity) {
    size_t size= 1u << capacity;
    resize(size);
}

void Transposition::resize(size_t capa) {
    capacity = 1u<<capa;
    entries = std::make_unique<Cluster[]>(capacity);
    clear();
}

size_t Transposition::getCapacity() {
    return capacity;
}

uint32_t Transposition::getHashHits() {
    return hashHit;
}

void Transposition::clear() {
    hashHit = 0;
    std::fill(entries.get(), entries.get()+capacity, Cluster{});
}

void Transposition::storeHash(Value value, uint64_t key, Flag flag, uint8_t depth, Move tt_move) {
    const auto index = (key) & (getCapacity() - 1u);
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
            cluster[i].bestMove = tt_move;
            cluster[i].value = value;
            cluster[i].key = lock;
            cluster[i].age = age_counter;
            return;
        }
    }
    cluster[0].depth = depth;
    cluster[0].flag = flag;
    cluster[0].bestMove = tt_move;
    cluster[0].value = value;
    cluster[0].key = lock;
    cluster[0].age = age_counter;
}

bool Transposition::findHash(uint64_t key, NodeInfo &info) {
    const auto index = key & (getCapacity() - 1u);
    const uint32_t currKey = key >> 32u;
    for (int i = 0; i < bucket_size; ++i) {
        if (this->entries[index][i].key == currKey) {
            info.tt_move = this->entries[index][i].bestMove;
            info.depth = this->entries[index][i].depth;
            info.flag = this->entries[index][i].flag;
            info.score = this->entries[index][i].value;
            return true;
        }
    }
    return false;
}

