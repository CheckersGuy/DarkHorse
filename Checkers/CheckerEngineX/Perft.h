
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
#include <Thread.h>
#include <atomic>
namespace Perft {

    struct SplitPoint {
        Position pos;
        uint8_t depth;
    };

    struct Thread : Lockable {

        static constexpr size_t MAX_SPLITPOINTS = 4;

        std::vector<std::unique_ptr<Thread>> &pool;
        std::thread local_thread;
        std::array<SplitPoint, MAX_SPLITPOINTS> split_points;
        size_t num_split_points{0};
        std::atomic<bool> stop{false};

        void idle_loop();

        std::optional<SplitPoint> find_work();

        bool has_work();

        std::optional<SplitPoint> get_split_point();

        void start_thread();

    };


    struct Entry {
        uint32_t WP, BP, K;
        uint64_t nodes;
        int8_t depth;

        Entry() = default;

        Entry(Position pos, int depth, uint64_t nodes) : WP(pos.WP), BP(pos.BP), K(pos.K), depth(depth), nodes(nodes) {}
    };


    using Cluster = std::array<Entry, 2>;

    class Table {

    private:
        size_t capacity;
        std::vector<Cluster> entries;
    public:
        Table(uint32_t capacity) : capacity(capacity) {
            set_capacity(capacity);
        }

        Table() = default;

        void clear();

        size_t get_capacity();

        void set_capacity(size_t capacity);

        void set_capacity(std::string capa_string);

        size_t probe(Position pos, int depth);

        void store(Position pos, int depth, uint64_t nodes);

    };

    extern Table table;

    uint64_t perft_check(Position &pos, int depth);


    void perft_check(Position &pos, int depth, PerftCallBack &call_back);

    struct MoveReceiver {
        PerftCallBack &call_back;
        Position &pos;
        int depth;

        template<MoveType type>
        inline void visit(uint32_t &maske, uint32_t &next) {
            Move mv{maske, next};
            Position copy = pos;
            copy.make_move(mv);
            Zobrist::update_zobrist_keys(copy, mv);
            perft_check(copy, depth - 1, call_back);
        };

        template<MoveType type>
        inline void visit(uint32_t &from, uint32_t &to, uint32_t &captures) {
            Move mv{from, to, captures};
            Position copy = pos;
            copy.make_move(mv);
            Zobrist::update_zobrist_keys(copy, mv);
            perft_check(copy, depth - 1, call_back);
        };

    };

}
#endif //CHECKERENGINEX_PERFT_H

