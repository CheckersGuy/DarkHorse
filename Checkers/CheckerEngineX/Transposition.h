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
    Value value{0};//4 bytes
    Move bestMove;//12 bytes
    uint32_t key{0u};//4 bytes
    uint32_t age{100000u};// 4 bytes
    Flag flag{Flag::None};//1 byte
    uint8_t depth{0};//1 byte
    //padding
    //total of 4+12+4+4+1+1 = 26 bytes !!!
    //horrible, would like a total of 16 bytes one again
};
constexpr size_t bucket_size = 4;


struct Cluster {
    std::array<Entry, bucket_size> ent;
};

class Transposition {

public:
    size_t hashHit{0u};
    uint32_t age_counter{0};
    size_t capacity{0};
    std::unique_ptr<Cluster[]> entries;

public:
    explicit Transposition(size_t length);

    Transposition() = default;

    size_t get_capacity() const;

    uint32_t get_hash_hits() const;

    void clear();

    void resize(size_t capa);

    void store_hash(Value value, uint64_t key, Flag flag, uint8_t depth, Move tt_move);

    bool find_hash(uint64_t key, NodeInfo &info) const;
};

extern Transposition TT;


#endif //CHECKERSTEST_TRANSPOSITION_H