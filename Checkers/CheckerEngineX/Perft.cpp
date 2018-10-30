
//
// Created by Robin on 14.01.2018.
//

#include "Perft.h"
#include "GameLogic.h"

namespace Perft {


    ThreadPool::ThreadPool(int th, SearchGlobal *global) {
        this->numThreads = th;
        this->global = global;
        for (int i = 0; i < th; ++i) {
            this->threads.push_back(std::thread(idleLoop, this));
        }
    }

    void ThreadPool::waitAll() {
        while (global->workCounter != 0);
    }

    void ThreadPool::join() {
        for (std::thread &current : threads) {
            if (current.joinable()) {
                current.join();
            }
        }
    }

    uint64_t ThreadPool::search(Board &board, int depth) {
        uint64_t cacheNodes = 0;
        if (depth >= 2) {
            cacheNodes = cache.findEntry(*board.getPosition(), depth);
        }
        if (cacheNodes > 0) {
            return cacheNodes;
        }


        uint64_t nodes = 0;
        MoveListe myListe;
        getMoves(board, myListe);
        if (depth == 1) {
            return myListe.length();
        }
        bool split = false;
        for (int i = 0; i < myListe.length(); ++i) {
            if (depth >= 7 && global->workCounter <= numThreads && i < myListe.length() - 1) {
                board.makeMove(myListe[i]);
                global->addWork(*board.getPosition(), depth - 1);
                board.undoMove();
                split = true;
                continue;
            }

            board.makeMove(myListe.liste[i]);
            nodes += search(board, depth - 1);
            board.undoMove();
        }

        if (depth >= 2 && !split) {
            cache.storeEntry(*board.getPosition(), depth, nodes);
        }

        return nodes;
    }

    void idleLoop(ThreadPool *pool) {
        while (true) {
            if (pool->global->condVar == false)
                break;

            if (pool->global->workCounter == 0)
                break;

            pool->global->mutex.lock();
            if (!pool->global->work.empty()) {
                Work current = pool->global->work.front();
                pool->global->work.pop();
                Board next;
                next.pStack[next.pCounter] = current.pos;
                pool->global->mutex.unlock();
                pool->global->workCounter--;
                uint64_t nodes = pool->search(next, current.depth);
                pool->global->nodeCounter += nodes;
                std::cout << "Global:" << pool->global->nodeCounter << std::endl;

            } else {
                pool->global->mutex.unlock();
            }


        }
    }

    void SearchGlobal::update(uint64_t nodes) {
        workCounter--;
        this->nodeCounter += nodes;
    }

    void SearchGlobal::addWork(Position pos, int depth) {
        mutex.lock();
        workCounter++;
        work.push(Work(pos, depth));
        mutex.unlock();
    }

    SearchGlobal::SearchGlobal() : condVar(true), workCounter(0) {}


    void printTree(Board &board, int depth) {
        if (depth == 0)
            return;

        MoveListe myListe;
        getMoves(board, myListe);


        for (int i = 0; i < myListe.length(); ++i) {
            board.makeMove(myListe.liste[i]);
            board.getPosition()->printPosition();
            std::cout << "\n";
            printTree(board, depth - 1);
            std::cout << "\n";
            board.undoMove();
        }
    }

    uint64_t benchMark(Board &board, int depth) {

        uint64_t cacheNodes = 0;
        if (depth >= 2) {
            cacheNodes = cache.findEntry(*board.getPosition(), depth);
        }
        if (cacheNodes > 0) {
            return cacheNodes;
        }

        uint64_t nodes = 0;
        MoveListe myListe;
        getMoves(board, myListe);
        if (depth == 1) {
            return myListe.length();
        }

        for (int i = 0; i < myListe.length(); ++i) {
            board.makeMove(myListe.liste[i]);
            nodes += benchMark(board, depth - 1);
            board.undoMove();
        }

        if (depth >= 2) {
            cache.storeEntry(*board.getPosition(), depth, nodes);
        }
        return nodes;
    }

    Cache::Cache(uint32_t capa) : capactiy(1 << capa) {
        this->clusters = new Cluster[capactiy];
        memset(clusters, 0, sizeof(Cluster) * capactiy);

    }

    uint64_t Cache::findEntry(Position &pos, int depth) {
        const uint32_t index = static_cast<uint32_t >(pos.key) & (capactiy - 1);
        Cluster current = clusters[index];
        if (current.entries[1].depth == depth && pos == current.entries[1].pos) {
            return current.entries[1].nodes;
        }
        if (current.entries[0].depth == depth && pos == current.entries[0].pos) {
            return current.entries[0].nodes;
        }
        return 0;
    }

    void Cache::initialize(uint32_t capa) {
        this->clusters = new Cluster[1 << capa];
        this->capactiy = 1 << capa;
        memset(clusters, 0, sizeof(Cluster) * capactiy);

    }

    void Cache::storeEntry(Position pos, int depth, uint64_t nodes) {
        const uint32_t index = static_cast<uint32_t >(pos.key) & (capactiy - 1);
        Entry next;
        next.pos = pos;
        next.depth = depth;
        next.nodes = nodes;
        if (next.depth > clusters[index].entries[0].depth) {
            clusters[index].entries[0] = next;
        } else {
            clusters[index].entries[1] = next;
        }
    }

    Cache cache;

}
