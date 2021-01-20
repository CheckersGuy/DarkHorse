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

struct NodeInfo {
    uint8_t move_index {Move_Index_None};
    Value score{NONE};
    uint8_t depth{0};
    uint8_t flag{Flag::None};
};

//given a cache-line of 64 bytes, we can efficiently use a bucket-size of 4 !
struct Entry {
    //total of 16 bytes
    Value value{NONE};
    uint32_t key{0u};
    uint32_t age{100000u}; //age
    Flag flag{Flag::None};
    uint8_t depth{0};
    uint8_t bestMove{Move_Index_None};
    uint8_t padding; //padding
};
constexpr size_t bucket_size = 4;

using Cluster = std::array<Entry, bucket_size>;

class Transposition {

public:
    size_t length{0u};
    size_t capacity{0u};
    size_t hashHit{0u};
    uint32_t age_counter{0};
    std::unique_ptr<Cluster[]> entries;

public:
    explicit Transposition(size_t length);

    Transposition() = default;

    uint32_t getLength();

    double getFillRate();

    uint32_t getCapacity();

    uint32_t getHashHits();

    void clear();

    void resize(size_t capa);

    void storeHash(Value value, const Position &pos, Flag flag, uint8_t depth, uint32_t move_index);

    bool findHash(uint64_t key, NodeInfo &info);
};


extern Transposition TT;

#endif //CHECKERSTEST_TRANSPOSITION_H