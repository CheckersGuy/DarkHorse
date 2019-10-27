//
// Created by Robin on 13.06.2017.
//


#include "Transposition.h"
#include <string.h>

Transposition TT;

Transposition::Transposition(uint32_t capacity) {
    entries = std::make_unique<Cluster[]>(1<<capacity);
    this->capacity = 1u << (capacity);
    memset(this->entries.get(), 0, 2 * sizeof(Entry) * this->capacity);
}

void Transposition::resize(uint32_t capa) {
    this->entries =std::make_unique<Cluster[]>(1u<<capa);
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


void Transposition::storeHash(Value value, uint64_t key, Flag flag, uint8_t depth, Move move) {
    this->length++;
    assert(value.isEval());
    const uint32_t index = (key) & (this->capacity - 1);
    assert(index < capacity);
    Cluster &cluster = this->entries[index];
    if (depth > cluster[1].getDepth()) {
        cluster[1].depth = depth;
        cluster[1].flag = flag;
        cluster[1].bestMove = move;
        cluster[1].value = value;
        cluster[1].key = static_cast<uint32_t >(key >> 32u);
        return;
    }
    cluster[0].depth = depth;
    cluster[0].flag = flag;
    cluster[0].value = value;
    cluster[0].key = static_cast<uint32_t >(key >> 32u);
    cluster[0].bestMove = move;
}

void Transposition::findHash(const uint64_t key, int depth, int *alpha, int *beta, NodeInfo &info) {
    assert(info.value.isInRange(-INFINITE, INFINITE));
    const uint32_t index = key & (this->capacity - 1);

    auto currKey = static_cast<uint32_t >(key >> 32u);
    for (int i = 0; i <= 1; ++i) {
        if (this->entries[index][i].key == currKey) {
            if (this->entries[index][i].getDepth() >= depth) {
                if (this->entries[index][i].getFlag() == TT_EXACT) {
                    (*alpha) = this->entries[index][i].value.value;
                    (*beta) = this->entries[index][i].value.value;
                } else if (this->entries[index][i].getFlag() == TT_LOWER) {
                    (*alpha) = std::max((*alpha), this->entries[index][i].value.value);
                } else if (this->entries[index][i].getFlag() == TT_UPPER) {
                    (*beta) = std::min((*beta), this->entries[index][i].value.value);
                }
                this->hashHit++;
            }
            info.move = this->entries[index][i].bestMove;
            info.value = this->entries[index][i].getValue();
            info.depth = this->entries[index][i].getDepth();
            info.flag = this->entries[index][i].getFlag();

            break;

        }
    }

}

