
//
// Created by Robin on 14.01.2018.
//

#ifndef CHECKERENGINEX_PERFT_H
#define CHECKERENGINEX_PERFT_H

#include "Position.h"
#include <mutex>
#include <atomic>
#include <deque>
#include "MGenerator.h"
#include "SMPLock.h"
#include <optional>
#include <vector>
#include <deque>
#include <thread>
#include <cstring>

namespace Perft {

    struct Entry {
        Position pos;
        int depth;
        uint64_t nodes;

        Entry() = default;

        Entry(Position pos, int depth, uint64_t nodes) : pos(pos), depth(depth), nodes(nodes) {}
    };

    struct Cluster {
        Entry entries[2];
    };


    class Table {

    private:
        uint32_t capacity;
        std::unique_ptr<Cluster[]> entries;
    public:
        Table(uint32_t capacity) : capacity(capacity), entries(std::make_unique<Cluster[]>(capacity)) {
            std::memset(entries.get(), 0, sizeof(Entry) * 2 * capacity);
        }

        Table() = default;

        uint64_t getCapacity();

        void setCapacity(uint32_t capacity);

        std::optional<uint64_t> probe(Position pos, int depth);

        void store(Position pos, int depth, uint64_t nodes);

    };

    extern Table table;

    uint64_t perftCheck(Position &pos, int depth);
}
#endif //CHECKERENGINEX_PERFT_H

