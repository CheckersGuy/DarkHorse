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


struct NodeInfo {
    Move move;
    Value score;
    uint8_t depth ;
    uint8_t  flag{Flag::None} ;
};


struct Entry {
    Move bestMove;
    Value value ;
    uint32_t key ;
    Flag flag;
    uint8_t depth;
    uint32_t getKey();

    Value getValue();

    uint8_t getFlag();

    uint8_t getDepth();

};




inline uint32_t Entry::getKey() {
    return this->key;
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

using Cluster=Entry[2];

class Transposition {

public:
    uint32_t length = 0;
    uint32_t capacity = 0;
    uint32_t hashHit = 0;
    std::unique_ptr<Cluster[]>entries;

public:
    explicit Transposition(uint32_t length);

    Transposition() = default;

    uint32_t getLength();

    double getFillRate();

    uint32_t getCapacity();

    uint32_t getHashHits();

    void clear();

    void resize(uint32_t capa);

    void storeHash(Value value, const Position&pos, Flag flag, uint8_t depth, Move move);

    bool findHash(const Position&pos, NodeInfo &info);
};


extern Transposition TT;

#endif //CHECKERSTEST_TRANSPOSITION_H