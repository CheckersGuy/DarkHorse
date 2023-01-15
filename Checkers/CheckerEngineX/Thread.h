//
// Created by robin on 30.01.22.
//

#ifndef READING_THREAD_H
#define READING_THREAD_H

#include <mutex>
#include <thread>
#include "Position.h"

struct Lockable {
    std::mutex mutex;

    void aquire();

    void release();
};

struct SplitPoint : public Lockable {
    SplitPoint *parent;
    Position pos;
    int depth;


};


class Thread {
    std::thread thread;
    size_t num_nodes{0ull};

};


#endif //READING_THREAD_H
