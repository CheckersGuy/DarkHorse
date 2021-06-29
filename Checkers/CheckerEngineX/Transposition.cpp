//
// Created by Robin on 13.06.2017.
//


#include "Transposition.h"

Transposition TT;

Transposition::Transposition(size_t capacity) {
    size_t size = 1u << capacity;
    resize(size);
}

void Transposition::resize(size_t capa) {
    capacity = 1u << capa;
    entries = std::make_unique<Cluster[]>(capacity);
    clear();
}

size_t Transposition::getCapacity() const {
    return capacity;
}

uint32_t Transposition::getHashHits() const {
    return hashHit;
}


void Transposition::clear() {
    hashHit = 0;
    std::fill(entries.get(), entries.get() + capacity, Cluster{});
}

void Transposition::storeHash(Value value, uint64_t key, Flag flag, uint8_t depth, Move tt_move) {
    const auto index = (key) & (getCapacity() - 1u);
    Cluster &cluster = this->entries[index];
    const uint32_t lock = (key >> 32u);


    for (auto i = 1; i < bucket_size; ++i) {
        if (cluster.ent[i].key == lock) {
            cluster.ent[i].age = age_counter;
        }
        const auto age_entry = age_counter - cluster.ent[i].age;

        if (cluster.ent[i].depth < depth || (age_entry != 0)) {
            cluster.ent[i].depth = depth;
            cluster.ent[i].flag = flag;
            cluster.ent[i].bestMove = tt_move;
            cluster.ent[i].value = value;
            cluster.ent[i].key = lock;
            cluster.ent[i].age = age_counter;
            return;
        }
    }
    cluster.ent[0].depth = depth;
    cluster.ent[0].flag = flag;
    cluster.ent[0].bestMove = tt_move;
    cluster.ent[0].value = value;
    cluster.ent[0].key = lock;
    cluster.ent[0].age = age_counter;
}

bool Transposition::findHash(uint64_t key, NodeInfo &info) const {
    const auto index = key & (getCapacity() - 1u);
    const uint32_t currKey = key >> 32u;
    for (int i = 0; i < bucket_size; ++i) {
        if (this->entries[index].ent[i].key == currKey) {
            info.tt_move = this->entries[index].ent[i].bestMove;
            info.depth = this->entries[index].ent[i].depth;
            info.flag = this->entries[index].ent[i].flag;
            info.score = this->entries[index].ent[i].value;
            return true;
        }
    }
    return false;
}

