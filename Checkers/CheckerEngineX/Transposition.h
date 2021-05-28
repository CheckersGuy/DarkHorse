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
    Move tt_move;
    Value score{0};
    uint8_t depth{0};
    uint8_t flag{Flag::None};
};

struct Entry {
    //total of 16 bytes
    Value value{0};
    Move bestMove;
    uint32_t key{0u};
    uint32_t age{100000u}; //age
    Flag flag{Flag::None};
    uint8_t depth{0};
};
constexpr size_t bucket_size = 4;

using Cluster = std::array<Entry, bucket_size>;

class Transposition {

public:
    size_t hashHit{0u};
    uint32_t age_counter{0};
    size_t capacity{0};
    std::unique_ptr<Cluster[]> entries;

public:
    explicit Transposition(size_t length);

    Transposition() = default;

    size_t getCapacity();

    uint32_t getHashHits();

    void clear();

    void resize(size_t capa);

    void storeHash(Value value, uint64_t key, Flag flag, uint8_t depth, Move tt_move);

    bool findHash(uint64_t key, NodeInfo &info);
};

extern Transposition TT;


#endif //CHECKERSTEST_TRANSPOSITION_H