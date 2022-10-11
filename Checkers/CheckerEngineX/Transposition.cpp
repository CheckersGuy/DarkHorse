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

size_t Transposition::get_capacity() const {
    return capacity;
}

uint32_t Transposition::get_hash_hits() const {
    return hashHit;
}


void Transposition::clear() {
    hashHit = 0;
    age_counter =0;
    std::fill(entries.get(), entries.get() + capacity, Cluster{});
}

void Transposition::store_hash(Value value, uint64_t key, Flag flag, uint8_t depth, uint8_t tt_index) {
 const auto index = (key) & (get_capacity() - 1u);
    Cluster &cluster = this->entries[index];
    const uint32_t lock = (key >> 32u);


    for (auto i = 1; i < bucket_size; ++i) {
        if (cluster.ent[i].key == lock) {
            cluster.ent[i].depth = depth;
            cluster.ent[i].flag = flag;
            cluster.ent[i].move_index = tt_index;
            cluster.ent[i].value = value;
            cluster.ent[i].age = age_counter;
            return;
        }
        const auto age_entry = age_counter - cluster.ent[i].age;


        if (cluster.ent[i].flag == Flag::None ||cluster.ent[i].depth < depth || (age_entry != 0)) {
            cluster.ent[i].depth = depth;
            cluster.ent[i].flag = flag;
            cluster.ent[i].move_index = tt_index;
            cluster.ent[i].value = value;
            cluster.ent[i].key = lock;
            cluster.ent[i].age = age_counter;
            return;
        }
    }
    cluster.ent[0].depth = depth;
    cluster.ent[0].flag = flag;
    cluster.ent[0].move_index = tt_index;
    cluster.ent[0].value = value;
    cluster.ent[0].key = lock;
    cluster.ent[0].age = age_counter;
}

bool Transposition::find_hash(uint64_t key, NodeInfo &info) const {
    const auto index = key & (get_capacity() - 1u);
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

