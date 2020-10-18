//
// Created by Robin on 13.06.2017.
//


#include "Transposition.h"
#include <string.h>

Transposition TT;

Transposition::Transposition(uint32_t capacity) {
    entries = std::make_unique<Cluster[]>(1 << capacity);
    this->capacity = 1u << (capacity);
    memset(this->entries.get(), 0, 2 * sizeof(Entry) * this->capacity);
}

void Transposition::resize(uint32_t capa) {
    this->entries = std::make_unique<Cluster[]>(1u << capa);
    this->capacity = 1u << capa;
    memset(this->entries.get(), 0, 2 * sizeof(Entry) * this->capacity);
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
    memset(this->entries.get(), 0, 2 * sizeof(Entry) * this->capacity);
}


double Transposition::getFillRate() {
    return ((double) length) / capacity;
}


void Transposition::storeHash(Value value, const Position &pos, Flag flag, uint8_t depth, Move move) {
    this->length++;
    const uint32_t index = (pos.key) & (this->capacity - 1);
    Cluster &cluster = this->entries[index];
    if (depth > cluster[1].depth) {
        cluster[1].depth = depth;
        cluster[1].flag = flag;
        cluster[1].bestMove = move;
        cluster[1].value = value;
        cluster[1].key = static_cast<uint32_t >(pos.key >> 32u);
        return;
    }
    cluster[0].depth = depth;
    cluster[0].flag = flag;
    cluster[0].value = value;
    cluster[0].key = static_cast<uint32_t >(pos.key >> 32u);
    cluster[0].bestMove = move;
}

bool Transposition::findHash(uint64_t key, NodeInfo &info) {
    const uint32_t index = key & (this->capacity - 1);
    auto currKey = static_cast<uint32_t >(key>> 32u);
    for (int i = 0; i <= 1; ++i) {
        if (this->entries[index][i].key == currKey) {
            info.move = this->entries[index][i].bestMove;
            info.depth = this->entries[index][i].depth;
            info.flag = this->entries[index][i].flag;
            info.score = this->entries[index][i].value;
            return true;
        }
    }

    return false;
}

