
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

    void Table::clear() {
        Cluster clust{};
        std::fill(entries.begin(),entries.end(),clust);
    }

    void Table::setCapacity(uint32_t capacity) {
        this->capacity = capacity;
        entries.resize(capacity);
        clear();
    }

    std::optional<uint64_t> Table::probe(Position pos, int depth) {
        auto key = static_cast<uint32_t >(pos.key >> 32u);
        const uint32_t index = (key) & (this->capacity - 1);
        if (entries[index][0].pos == pos && entries[index][0].depth == depth) {
            return std::make_optional(entries[index][0].nodes);
        } else if (entries[index][1].pos == pos && entries[index][1].depth == depth) {
            return std::make_optional(entries[index][1].nodes);
        }
        return std::nullopt;
    }

    void Table::store(Position pos, int depth, uint64_t nodes) {
        auto key = static_cast<uint32_t >(pos.key >> 32u);
        const uint32_t index = (key) & (this->capacity - 1u);
        if (depth > entries[index][0].depth) {
            entries[index][0].pos = pos;
            entries[index][0].depth = depth;
            entries[index][0].nodes = nodes;
        } else {
            entries[index][1].pos = pos;
            entries[index][1].depth = depth;
            entries[index][1].nodes = nodes;
        };
    }

    uint64_t perftCheck(Position& pos, int depth) {
        MoveListe liste;
        get_moves(pos, liste);
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
            copy.make_move(liste[i]);
            Zobrist::update_zobrist_keys(copy, liste[i]);
            counter+=perftCheck(copy,depth-1);
        }

        table.store(pos, depth, counter);
        return counter;
    }
}
