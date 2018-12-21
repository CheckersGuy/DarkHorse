
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
#include "SMPLock.h"


namespace Perft {

    uint64_t perftCount(int depth, int threads);

    uint64_t perftCount(int depth);


    struct Work {
        Position pos;
        int depth = 0;
    };

    struct Entry{
        constexpr static int DEPTH_ENTRY=1;
        constexpr static int FILL_ENTRY=0;
        int depth=0;
        Position pos;
        uint64_t nodes=0;

        bool operator ==(const Entry& other);
    };

    inline bool Entry::operator==(const Perft::Entry &other) {
        return this->pos==other.pos;
    }

    struct Bucket{
        Entry entries[2];
        SMPLock lock;
    };

    class PerftTable{

    private:
        Bucket* entries;
        std::size_t  capacity;
    public:

        PerftTable(uint64_t capa):capacity(capa){
            entries = new Bucket[capacity];
            this->capacity=capa;
            for(std::size_t i =0;i<capa;++i){
                Position empty;
                entries[0].entries[0].pos=empty;
                entries[0].entries[1].pos=empty;
            }
        }

        ~PerftTable(){
            delete[] entries;
        }

        Entry* findEntry(const Position pos,int depth,uint64_t key);



        void storeEntry(Position pos, int depth,uint64_t nodes,const uint64_t key);

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

