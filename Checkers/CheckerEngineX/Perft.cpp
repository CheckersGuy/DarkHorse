
//
// Created by Robin on 14.01.2018.
//

#include "Perft.h"


namespace Perft {


    Table table;

    uint64_t Table::get_capacity() {
        return capacity;
    }

    void Table::set_capacity(std::string capa_string) {
        auto length = std::max(0ull, capa_string.length() - 2ull);
        std::string size_string = capa_string.substr(0, length);
        std::string unit_string = capa_string.substr(length, 2);

        size_t size_in_bytes = 1;
        if (size_string.empty() || unit_string.empty()) {
            set_capacity(0);
            return;
        }
        auto result = std::find_if(size_string.begin(), size_string.end(), [](char c) {
            return !std::isdigit(c);
        });
        if (result == capa_string.end()) {
            set_capacity(0);
            return;
        }
        std::transform(unit_string.begin(), unit_string.end(), unit_string.begin(), tolower);
        size_in_bytes = std::stoi(size_string);
        if (unit_string == "b") {
            size_in_bytes *= 1ull;
        } else if (unit_string == "kb") {
            size_in_bytes *= 1000ull;
        } else if (unit_string == "mb") {
            size_in_bytes *= 1000000ull;
        } else if (unit_string == "gb") {
            size_in_bytes *= 1000000000ull;
        } else if (unit_string == "tb") {
            size_in_bytes *= 1000000000000ull;
        }

        //we round down to the largest number of entries which uses less than size_in_bytes

        size_t bytes_per_entry = sizeof(Cluster);
        size_t num_entries = size_in_bytes / bytes_per_entry;
        set_capacity(num_entries);
        std::cout<<"entries: "<<num_entries<<std::endl;
    }

    void Table::clear() {
        Cluster clust{};
        std::fill(entries.begin(), entries.end(), clust);
    }

    void Table::set_capacity(uint32_t capacity) {
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

    uint64_t perft_check(Position &pos, int depth) {
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
        for (int i = 0; i < liste.length(); ++i) {
            Position copy = pos;
            copy.make_move(liste[i]);
            Zobrist::update_zobrist_keys(copy, liste[i]);
            counter += perft_check(copy, depth - 1);
        }

        table.store(pos, depth, counter);
        return counter;
    }
}
