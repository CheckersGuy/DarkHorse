
//
// Created by Robin on 14.01.2018.
//

#include "Perft.h"
#include "BoardFactory.h"

namespace Perft {

    Table table;

    uint64_t Table::getCapacity() {
        return capacity;
    }

    void Table::setCapacity(uint32_t capacity) {
        this->capacity = capacity;
        entries = std::make_unique<Cluster[]>(capacity);
        std::memset(entries.get(), 0, sizeof(Entry) * 2 * capacity);
    }

    std::optional<uint64_t> Table::probe(Position pos, int depth) {
        std::optional<uint64_t> returnValue;
        uint32_t key = static_cast<uint32_t >(pos.key >> 32);
        const uint32_t index = (key) & (this->capacity - 1);
        entries[index].lock.lock();
        if (entries[index].entries[1].pos == pos && entries[index].entries[1].depth == depth) {
            returnValue = entries[index].entries[1].nodes;
        } else if (entries[index].entries[0].pos == pos && entries[index].entries[0].depth == depth) {
            returnValue = entries[index].entries[0].nodes;
        }
        entries[index].lock.unlock();
        return returnValue;
    }

    void Table::store(Position pos, int depth, uint64_t nodes) {
        uint32_t key = static_cast<uint32_t >(pos.key >> 32);
        const uint32_t index = (key) & (this->capacity - 1);
        entries[index].lock.lock();
        if (depth>entries[index].entries[0].depth ) {
            entries[index].entries[0].pos = pos;
            entries[index].entries[0].depth = depth;
            entries[index].entries[0].nodes = nodes;
        } else {
            entries[index].entries[1].pos = pos;
            entries[index].entries[1].depth = depth;
            entries[index].entries[1].nodes = nodes;
        };
        entries[index].lock.unlock();
    }

    uint64_t perftCheck(Board &board, int depth) {
        MoveListe liste;
        getMoves(*board.getPosition(), liste);
        if (depth == 1) {
            return liste.length();
        }
        uint64_t counter = 0;
        for (Move m : liste) {
            board.makeMove(m);
            counter += perftCheck(board, depth - 1);
            board.undoMove();
        }
        return counter;
    }

    uint64_t ThreadPool::perftCount(Board &board, int depth) {
        MoveListe liste;
        getMoves(*board.getPosition(), liste);
        if (depth == 1) {
            return liste.length();
        }
        std::optional<uint64_t> nodes = table.probe(*board.getPosition(), depth);
        if (nodes.has_value()) {
            return nodes.value();
        }

        uint64_t counter = 0;
        for (Move m : liste) {
            board.makeMove(m);
            counter += perftCount(board, depth - 1);
            board.undoMove();
        }

        table.store(*board.getPosition(), depth, counter);

        return counter;

    }

    void ThreadPool::waitAll() {
        for (auto &th : workers)
            th.join();
    }

    uint64_t ThreadPool::getNodeCounter() {
        return nodeCounter;
    }

    void ThreadPool::splitLoop(Board &board, int depth, int startDepth) {

        MoveListe liste;
        getMoves(*board.getPosition(), liste);

        for (size_t i = 0; i < liste.length(); ++i) {
            if (workCounter >= 200) {
                i--;
                continue;
            }
            board.makeMove(liste[i]);
            if (depth == splitDepth) {
                Position copy = *board.getPosition();
                SplitPoint point;
                point.depth = depth - 1;
                point.pos = copy;
                myMutex.lock();
                splitPoints.emplace_back(point);
                workCounter++;
                myMutex.unlock();
                board.undoMove();
                continue;

            }
            splitLoop(board, depth - 1, startDepth);
            board.undoMove();
        }
        if (depth == startDepth)
            search = !search;

    }

    void ThreadPool::startThreads() {
        for (size_t i = 0; i < numThreads; ++i) {
            workers.emplace_back(std::thread(idleLoop, this));
        }
    }

    void ThreadPool::setSplitDepth(int splitDepth) {
        this->splitDepth=splitDepth;
    }

    void ThreadPool::idleLoop(ThreadPool *pool) {
        while (!pool->search || pool->workCounter > 0) {
            pool->myMutex.lock();
            std::optional<SplitPoint> splitPoint;
            if (!pool->splitPoints.empty()) {
                splitPoint = pool->splitPoints.front();
                pool->splitPoints.pop_front();
            }
            pool->myMutex.unlock();

            if (splitPoint.has_value()) {
                const int depth = splitPoint.value().depth;
                const Position pos = splitPoint.value().pos;
                Board board;
                BoardFactory::setUpPosition(board, pos);
                uint64_t nodes = pool->perftCount(board, depth);
                pool->workCounter--;
                pool->myMutex.lock();
                pool->nodeCounter += nodes;
                pool->myMutex.unlock();
            }
        }
    }

}
