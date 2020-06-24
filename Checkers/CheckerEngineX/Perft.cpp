
//
// Created by Robin on 14.01.2018.
//

#include "Perft.h"


namespace Perft {


    Table table;

    uint64_t Table::getCapacity() {
        return capacity;
    }

    void Table::setCapacity(std::string capa_string) {
        //specifying the ram to use
        //example "12mb" will use the largest power of
        //two less than 12mb and so on up to tb/TB for terabytes
        //of memory :P
        //Probably best to check
    }

    void Table::setCapacity(uint32_t capacity) {
        this->capacity = capacity;
        entries = std::make_unique<Cluster[]>(capacity);
        std::memset(entries.get(), 0, sizeof(Entry) * 2 * capacity);
    }

    std::optional<uint64_t> Table::probe(Position pos, int depth) {
        std::optional<uint64_t> returnValue;
        auto key = static_cast<uint32_t >(pos.key >> 32u);
        const uint32_t index = (key) & (this->capacity - 1);
        if (entries[index].entries[1].pos == pos && entries[index].entries[1].depth == depth) {
            returnValue = entries[index].entries[1].nodes;
        } else if (entries[index].entries[0].pos == pos && entries[index].entries[0].depth == depth) {
            returnValue = entries[index].entries[0].nodes;
        }
        return returnValue;
    }

    void Table::store(Position pos, int depth, uint64_t nodes) {
        auto key = static_cast<uint32_t >(pos.key >> 32u);
        const uint32_t index = (key) & (this->capacity - 1u);
        if (depth > entries[index].entries[0].depth) {
            entries[index].entries[0].pos = pos;
            entries[index].entries[0].depth = depth;
            entries[index].entries[0].nodes = nodes;
        } else {
            entries[index].entries[1].pos = pos;
            entries[index].entries[1].depth = depth;
            entries[index].entries[1].nodes = nodes;
        };
    }

    uint64_t perftCheck(Position& pos, int depth) {
        MoveListe liste;
        getMoves(pos, liste);
        if (depth == 1) {
            return liste.length();
        }
        uint64_t counter = 0;
        auto result = table.probe(pos, depth);
        if (result.has_value()) {
            return result.value();
        }
        for (int i=0;i<liste.length();++i) {
            Position copy=pos;
            copy.makeMove(liste[i]);
            Zobrist::doUpdateZobristKey(copy, liste[i]);
            counter+=perftCheck(copy,depth-1);
        }

        table.store(pos, depth, counter);
        return counter;
    }
}
