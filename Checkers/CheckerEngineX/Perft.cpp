
//
// Created by Robin on 14.01.2018.
//

#include "Perft.h"
#include "GameLogic.h"

namespace Perft {

    uint64_t perftCount(int depth,int threads) {
        PerftPool pool(threads-1);
        pool.setDepth(depth);
        pool.startThreads();
        PerftPool::idleLoop(&pool);
        pool.join();
        return pool.getPerftCount();
    }

    uint64_t perftCount(int depth) {
        return perftCount(depth,1);
    }

    uint64_t PerftPool::perftCount(Board &board, int depth) {
        uint64_t nodes = 0;
        MoveListe myListe;
        getMoves(*board.getPosition(), myListe);
        if (depth == 1) {
            return myListe.length();
        }
        for (int i = 0; i < myListe.length(); ++i) {
            if(depth>=10 && work.size()<=2*threads){
                myMutex.lock();
                board.makeMove(myListe[i]);
                Work next;
                next.depth=depth-1;
                next.pos=*board.getPosition();
                work.push(next);
                board.undoMove();
                myMutex.unlock();
                continue;
            }
            board.makeMove(myListe.liste[i]);
            nodes += perftCount(board, depth - 1);
            board.undoMove();
        }

        return nodes;
    }

    void PerftPool::join() {
        for (auto &th : myThreads) {
            if (!th.joinable())
                continue;
            th.join();
        }
    }

    PerftPool::~PerftPool() {
        join();
    }

    void PerftPool::setDepth(int depth) {
        this->depth = depth;
        Board board;
        BoardFactory::setUpStartingPosition(board);
        Work current;
        current.depth = depth;
        current.pos = *board.getPosition();
        work.push(current);
    }

    void PerftPool::idleLoop(Perft::PerftPool *pool) {

        while (!pool->work.empty()) {
            pool->myMutex.lock();
            Work current;
            if (!pool->work.empty()) {
                current = pool->work.front();
                pool->work.pop();
            }
            pool->myMutex.unlock();
            if (current.depth == 0)
                continue;
            Board board;
            BoardFactory::setUpPosition(board, current.pos);
            const uint64_t counter = pool->perftCount(board, current.depth);
            pool->nodeCounter += counter;
        }

    }

    uint64_t PerftPool::getPerftCount() {
        return nodeCounter;
    }

    void PerftPool::startThreads() {
        for (int i = 0; i < threads; ++i) {
            myThreads.emplace_back(std::thread(idleLoop, this));
        }
    }


}
