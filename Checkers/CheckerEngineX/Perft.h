
//
// Created by Robin on 14.01.2018.
//

#ifndef CHECKERENGINEX_PERFT_H
#define CHECKERENGINEX_PERFT_H

#include <mutex>
#include <queue>
#include <atomic>
#include "Board.h"
#include "BoardFactory.h"


namespace Perft {

    uint64_t perftCount(int depth, int threads);

    uint64_t perftCount(int depth);


    struct Work {
        Position pos;
        int depth = 0;
    };

    class PerftPool {
    private:
        std::atomic<uint64_t> nodeCounter;
        std::mutex myMutex;
        std::vector<std::thread> myThreads;
        std::queue<Work> work;
        int threads;
        int depth;

    public:

        PerftPool(int threads) : threads(threads), nodeCounter(0), depth(0) {

        }

        ~PerftPool();

        void startThreads();

        void join();

        static void idleLoop(PerftPool *pool);

        uint64_t perftCount(Board &board, int depth);

        uint64_t getPerftCount();

        void setDepth(int depth);


    };
}
#endif //CHECKERENGINEX_PERFT_H

