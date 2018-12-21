//
// Created by Robin on 13.06.2017.
//


#include "Transposition.h"
#include <string.h>

Transposition TT;

void Entry::printEncoding() {
    short copy = encoding;
    std::string output;
    while (encoding) {
        if (copy & 1 == 0) {
            output += "0";
        } else {
            output += "1";
        }
        copy = copy >> 1;
    }
    for (char i = output.length() - 1; i >= 0; i--) {
        std::cout << (int) output.at(i);
    }
    std::cout << std::endl;
}

Transposition::Transposition(uint32_t capacity) {
    entries = new Cluster[1 << capacity];
    this->capacity = 1 << (capacity);
    memset(this->entries, 0, 2 * sizeof(Entry) * this->capacity);
}

Transposition::~Transposition() {
    delete[]this->entries;
}

void Transposition::resize(uint32_t capa) {
    delete[] this->entries;
    this->entries = new Cluster[1 << capa];
    this->capacity = 1 << capa;
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
    memset(this->entries, 0, 2 * sizeof(Entry) * this->capacity);
}

void Transposition::incrementAgeCounter() {
    ageCounter++;
    ageCounter=ageCounter%Transposition::AGE_LIMIT;
}

uint32_t Transposition::getAgeCounter() {
    return ageCounter;
}


double Transposition::getFillRate() {
    return ((double) length) / capacity;
}


void Transposition::storeHash(Value value, uint64_t key, Flag flag, uint16_t depth, Move move) {
    this->length++;
    assert(value.isEval());
    const uint32_t index = (key) & (this->capacity - 1);
    Cluster *cluster = &this->entries[index];
    if (depth > cluster->entries[1].getDepth() || ageCounter !=cluster->entries[1].getAgeCounter()) {
        cluster->entries[1].setDepth(depth);
        cluster->entries[1].setFlag(flag);
        cluster->entries[1].setAgeCounter(ageCounter);
        cluster->entries[1].bestMove = move;
        cluster->entries[1].value = value;
        cluster->entries[1].key = static_cast<uint32_t >(key >> 32);
        return;
    }
    cluster->entries[0].setDepth(depth);
    cluster->entries[0].setFlag(flag);
    cluster->entries[0].setAgeCounter(ageCounter);
    cluster->entries[0].value = value;
    cluster->entries[0].key = static_cast<uint32_t >(key >> 32);
    cluster->entries[0].bestMove = move;
}

void Transposition::findHash(const uint64_t key, int depth, int *alpha, int *beta, NodeInfo &info) {

    assert(info.value.isInRange(-INFINITE, INFINITE));
    const uint32_t index = key & (this->capacity - 1);
    const uint32_t currKey = static_cast<uint32_t >(key >> 32);
    for (int i = 0; i <= 1; ++i) {
        if (this->entries[index].entries[i].key == currKey) {
            if (this->entries[index].entries[i].getDepth() >= depth) {
                if (this->entries[index].entries[i].getFlag() == TT_EXACT) {
                    (*alpha) = this->entries[index].entries[i].value.value;
                    (*beta) = this->entries[index].entries[i].value.value;
                } else if (this->entries[index].entries[i].getFlag() == TT_LOWER) {
                    (*alpha) = std::max((*alpha), this->entries[index].entries[i].value.value);
                } else if (this->entries[index].entries[i].getFlag() == TT_UPPER) {
                    (*beta) = std::min((*beta), this->entries[index].entries[i].value.value);
                }
                this->hashHit++;
            }
            info.move = this->entries[index].entries[i].bestMove;
            info.value = this->entries[index].entries[i].getValue();
            info.depth = this->entries[index].entries[i].getDepth();
            info.flag = this->entries[index].entries[i].getFlag();

            break;

        }
    }

}