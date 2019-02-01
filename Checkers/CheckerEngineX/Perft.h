
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

    struct Entry{
        Position pos;
        int depth;
        uint64_t nodes;

        Entry()=default;
        Entry(Position pos, int depth, uint64_t nodes):pos(pos),depth(depth),nodes(nodes){}
    };

    struct Cluster{
        Entry entries[2];
        SMPLock lock;
    };


    class Table{

    private:
        uint32_t capacity;
        std::unique_ptr<Cluster[]>entries;

    public:
        Table(uint32_t capacity):capacity(capacity),entries(std::make_unique<Cluster[]>(capacity)){
            std::memset(entries.get(),0,sizeof(Entry)*2*capacity);
        }

        Table()=default;

        uint64_t getCapacity();

        void setCapacity(uint32_t capacity);

        std::optional<uint64_t >probe(Position pos,int depth);

        void store(Position pos, int depth, uint64_t nodes);

    };


    struct SplitPoint {
        Position pos;
        int depth;
        SplitPoint(Position pos, int depth) : pos(pos), depth(depth) {}
        SplitPoint()=default;
    };



    uint64_t perftCheck(Board& board, int depth);

    class ThreadPool {
        using node_ptr=std::shared_ptr<SplitPoint>;

    public:
         int splitDepth=10;
        std::mutex myMutex;
        size_t numThreads;
        std::atomic<bool> search;
        std::deque<SplitPoint>splitPoints;
        std::vector<std::thread>workers;
        uint64_t nodeCounter=0;
        std::atomic<int>workCounter;
    public:


        ThreadPool(size_t numThreads) : numThreads(numThreads), search(false),workCounter(0) {}

        static void idleLoop(ThreadPool *pool);

        void startThreads();

        void waitAll();

        void setSplitDepth(int splitDepth);

        uint64_t perftCount(Board&board, int depth);

        void splitLoop(Board& board, int depth,int startDepth);

        uint64_t getNodeCounter();

    };

    extern Table table;

}
#endif //CHECKERENGINEX_PERFT_H

