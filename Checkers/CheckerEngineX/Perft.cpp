
//
// Created by Robin on 14.01.2018.
//

#include "Perft.h"


namespace Perft {


    Table table;

    size_t Table::get_capacity() {
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
    }

    void Table::clear() {
        Cluster clust{};
        std::fill(entries.begin(), entries.end(), clust);
    }

    void Table::set_capacity(size_t capacity) {
        this->capacity = capacity;
        entries.resize(capacity);
        clear();
    }

    size_t Table::probe(Position pos, int depth) {
        auto index = pos.key % capacity;
        if (entries[index][1].WP == pos.WP && entries[index][1].BP == pos.BP && entries[index][1].K == pos.K &&
            entries[index][1].depth == depth) {
            return entries[index][1].nodes;
        } else if (entries[index][0].WP == pos.WP && entries[index][0].BP == pos.BP && entries[index][0].K == pos.K &&
                   entries[index][0].depth == depth) {
            return entries[index][0].nodes;
        }
        return 0;
    }

    void Table::store(Position pos, int depth, uint64_t nodes) {
        auto index = pos.key % capacity;
        if (depth > entries[index][0].depth) {
            entries[index][0] = Entry(pos, depth, nodes);
        } else {
            entries[index][1] = Entry(pos, depth, nodes);
        };
    }

    uint64_t perft_check(Position &pos, int depth) {
        MoveListe liste;
        get_moves(pos, liste);
        if (depth == 1) {
            return liste.length();
        }
        uint64_t counter = 0;
        /*     auto result = table.probe(pos, depth);
             if (result != 0) {
                 return result;
             }*/
        for (int i = 0; i < liste.length(); ++i) {
            Position copy = pos;
            copy.make_move(liste[i]);
            Zobrist::update_zobrist_keys(copy, liste[i]);
            counter += perft_check(copy, depth - 1);
        }

        table.store(pos, depth, counter);
        return counter;
    }

    void perft_check(Position &pos, int depth, PerftCallBack &call_back) {
        if (depth == 1) {
            get_moves(pos, call_back);
            return;
        }

        /*    auto result = table.probe(pos, depth);
            if (result != 0) {
                call_back.num_nodes += result;
                return;
            }*/
        //uint64_t start_nodes = call_back.num_nodes;
        MoveReceiver receiver{call_back, pos, depth};
        get_moves(pos, receiver);
        //uint64_t nodes_searched = call_back.num_nodes - start_nodes;
        //table.store(pos, depth, nodes_searched);
        return;
    }


}


