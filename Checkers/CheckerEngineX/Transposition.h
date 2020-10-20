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
    uint8_t move_index {std::numeric_limits<uint8_t>::max()};
    Value score;
    uint8_t depth;
    uint8_t flag{Flag::None};
};

//given a cache-line of 64 bytes, we can efficiently use a bucket-size of 4 !
struct Entry {
    //total of 16 bytes
    Value value;
    uint32_t key;
    uint32_t p1; //padding
    Flag flag;
    uint8_t depth;
    uint8_t bestMove;
    uint8_t age; //padding
};
constexpr size_t bucket_size = 2;

using Cluster = std::array<Entry, bucket_size>;

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

    void storeHash(Value value, const Position &pos, Flag flag, uint8_t depth, uint8_t move_index);

    bool findHash(uint64_t key, NodeInfo &info);
};


extern Transposition TT;

#endif //CHECKERSTEST_TRANSPOSITION_H