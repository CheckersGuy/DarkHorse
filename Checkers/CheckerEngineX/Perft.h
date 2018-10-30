
//
// Created by Robin on 14.01.2018.
//

#ifndef CHECKERENGINEX_PERFT_H
#define CHECKERENGINEX_PERFT_H

#include "Board.h"
#include <atomic>
#include <mutex>
#include <queue>
#include <cstring>
#include <thread>

namespace Perft {


    struct Work {
        Position pos;
        int depth;

        Work(Position pos, int depth) : pos(pos), depth(depth) {};

        Work() = default;
    };

    struct SearchGlobal {
        std::mutex mutex;
        uint64_t nodeCounter = 0;
        std::atomic<bool> condVar;
        std::queue<Work> work;
        std::atomic<int> workCounter;

        SearchGlobal();

        void update(uint64_t nodes);

        void addWork(Position pos, int depth);
    };

    struct ThreadPool;

    void idleLoop(ThreadPool *global);


    struct ThreadPool {
        int numThreads;
        std::vector<std::thread> threads;
        SearchGlobal *global;

        ThreadPool(int th, SearchGlobal *global);

        ThreadPool() = default;

        uint64_t search(Board &board, int depth);

        void waitAll();

        void join();
    };


    struct Entry {
        Position pos;
        int depth;
        uint64_t nodes;

        Entry() = default;
    };


    void printTree(Board &board, int depth);

    uint64_t benchMark(Board &board, int depth);


    struct Cluster {
        Entry entries[2];
    };

    struct Cache {
        Cache() = default;

        Cluster *clusters;
        uint32_t capactiy;

        Cache(uint32_t capa);

        ~Cache() {
            delete[] clusters;
        }

        uint64_t findEntry(Position &pos, int depth);

        void storeEntry(Position pos, int depth, uint64_t nodes);

        void initialize(uint32_t capa);
    };


    extern Cache cache;


}
#endif //CHECKERENGINEX_PERFT_H

