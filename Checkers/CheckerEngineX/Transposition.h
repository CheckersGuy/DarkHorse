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
    uint8_t depth = 0;
    uint8_t  flag = 0;
    Move move;
};


struct Entry {
    Move bestMove;
    Value value ;
    uint32_t key ;
    uint8_t flag;
    uint8_t age;
    uint8_t depth;


    uint32_t getKey();

    Value getValue();

    uint8_t getFlag();

    uint8_t getAgeCounter();

    uint8_t getDepth();

};




inline uint32_t Entry::getKey() {
    return this->key;
}


inline uint8_t Entry::getAgeCounter() {
    return age;
}

inline Value Entry::getValue() {
    return this->value;
}

inline uint8_t Entry::getFlag() {
    return flag;
}


inline uint8_t Entry::getDepth() {
    return depth;
}

struct Cluster{
    Entry entries[2];
};

class Transposition {

    static constexpr uint8_t AGE_LIMIT = 250;

public:
    uint8_t ageCounter = 0;
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

    uint8_t getAgeCounter();

    void clear();

    void resize(uint32_t capa);

    void incrementAgeCounter();

    void storeHash(Value value, uint64_t key, Flag flag, uint8_t depth, Move move);

    void findHash(uint64_t key, int depth, int *alpha, int *beta, NodeInfo &info);
};


extern Transposition TT;

#endif //CHECKERSTEST_TRANSPOSITION_H