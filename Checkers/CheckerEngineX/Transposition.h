//
// Created by Robin on 13.06.2017.
//

#ifndef CHECKERSTEST_TRANSPOSITION_H
#define CHECKERSTEST_TRANSPOSITION_H

#include "types.h"
#include <string>
#include "Move.h"
#include "MoveListe.h"
#include <iostream>
#include <cstdint>


//Here will be some changes for the search branch
// in particular fractional Depth !!!
// if it doesnt have a huge performance penality
// I will keep it regardless



struct NodeInfo {
    Value value = 0;
    int depth = 0;
    uint16_t flag = 0;
    Move move;
};


struct Entry {
    uint16_t encoding = 0;
    Value value = 0;
    uint32_t key = 0;
    Move bestMove;


    void setDepth(const uint16_t depth);

    void setFlag(const uint16_t flag);

    void setAgeCounter(uint16_t flag);

    uint32_t getKey();

    Value getValue();

    uint16_t getEncoding();

    uint16_t getFlag();

    uint16_t getAgeCounter();

    uint16_t getDepth();

    void printEncoding();

    bool isEmpty();

};


inline bool Entry::isEmpty() {
    return encoding == 0;
}

inline uint16_t Entry::getEncoding() {
    return encoding;
}

inline uint32_t Entry::getKey() {
    return this->key;
}

inline void Entry::setAgeCounter(uint16_t flag) {
    this->encoding &= ~12;
    this->encoding |= flag << 2;
}

inline uint16_t Entry::getAgeCounter() {
    return (encoding & (12)) >> 2;
}

inline void Entry::setFlag(const uint16_t flag) {
    this->encoding &= ~3;
    this->encoding |= flag;
}

inline void Entry::setDepth(const uint16_t depth) {
    assert(depth < MAX_PLY);
    this->encoding &= ~16256;
    this->encoding |= depth << 7;
}

inline Value Entry::getValue() {
    return this->value;
}

inline uint16_t Entry::getFlag() {
    return encoding & 3;
}


inline uint16_t Entry::getDepth() {
    return (encoding & (16256)) >> 7;
}

struct Cluster {
    Entry entries[2];
};


class Transposition {

    static constexpr uint16_t AGE_LIMIT = 3;

private:
    uint16_t ageCounter = 0;
    uint32_t length = 0;
    uint32_t capacity = 0;
    uint32_t hashHit = 0;
    Cluster *entries;

public:
    Transposition(uint32_t length);

    Transposition() = default;

    ~Transposition();

    uint32_t getLength();

    double getFillRate();

    uint32_t getCapacity();

    uint32_t getHashHits();

    uint32_t getAgeCounter();

    void clear();

    void resize(uint32_t capa);

    void incrementAgeCounter();

    void storeHash(Value value, uint64_t key, Flag flag, uint16_t depth, Move move);

    void findHash(uint64_t key, int depth, int *alpha, int *beta, NodeInfo &info);
};


extern Transposition TT;

#endif //CHECKERSTEST_TRANSPOSITION_H