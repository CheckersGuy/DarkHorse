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
    uint8_t depth;
    uint8_t flag{Flag::None};
};


struct Entry {
    Move bestMove;
    Value value;
    uint32_t key;
    Flag flag;
    uint8_t depth;
};

using Cluster=std::array<Entry, 2>;

class Transposition {

public:
    uint32_t length{0u};
    uint32_t capacity{0u};
    uint32_t hashHit{0u};
    std::unique_ptr<Cluster[]> entries;

public:
    explicit Transposition(uint32_t length);

    Transposition() = default;

    uint32_t getLength();

    double getFillRate();

    uint32_t getCapacity();

    uint32_t getHashHits();

    void clear();

    void resize(uint32_t capa);

    void storeHash(Value value, const Position &pos, Flag flag, uint8_t depth, Move move);

    bool findHash(const Position &pos, NodeInfo &info);
};


extern Transposition TT;

#endif //CHECKERSTEST_TRANSPOSITION_H