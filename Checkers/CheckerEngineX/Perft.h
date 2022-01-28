
//
// Created by Robin on 14.01.2018.
//

#ifndef CHECKERENGINEX_PERFT_H
#define CHECKERENGINEX_PERFT_H

#include "Position.h"
#include <deque>
#include "MGenerator.h"
#include <optional>
#include <vector>
#include <deque>
#include <cstring>
namespace Perft {

    struct Entry {
        Position pos;
        int depth;
        uint64_t nodes;

        Entry() = default;

        Entry(Position pos, int depth, uint64_t nodes) : pos(pos), depth(depth), nodes(nodes) {}
    };


    using Cluster = std::array<Entry,2>;

    class Table {

    private:
        uint32_t capacity;
        std::vector<Cluster>entries;
    public:
        Table(uint32_t capacity) : capacity(capacity) {
            set_capacity(capacity);
        }

        Table() = default;

        void clear();

        uint64_t get_capacity();

        void set_capacity(uint32_t capacity);

        void set_capacity(std::string capa_string);

        std::optional<uint64_t> probe(Position pos, int depth);

        void store(Position pos, int depth, uint64_t nodes);

    };

    extern Table table;

    uint64_t perft_check(Position &pos, int depth);
}
#endif //CHECKERENGINEX_PERFT_H

