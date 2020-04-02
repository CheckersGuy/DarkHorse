//
// Created by robin on 4/2/19.
//

#ifndef CHECKERENGINEX_THREAD_H
#define CHECKERENGINEX_THREAD_H

//Will continue all the Thread-stuff (including threadpool)
//Needed for YBWC

#include <memory>
#include <thread>
#include <mutex>
#include <thread>
#include <condition_variable>

//Just some experimentation with
//the structure of the code I will be using
struct SplitPoint {

};


struct ThreadBase {
    static constexpr int MAX_SPLITPOINS = 6;
    using pointer = std::shared_ptr<SplitPoint>;

    std::array<pointer, MAX_SPLITPOINS> split_pointers;
    size_t num_points = 0ull;
    bool stop;
    std::mutex local_mutex;
    std::condition_variable cond_var;
    std::thread native_thread;


    void idleLoop();

    void stop_thread();

    void start_thread();

};

void ThreadBase::stop_thread() {
    std::lock_guard guard(local_mutex);
    stop = true;
    cond_var.notify_all();
}

void ThreadBase::idleLoop() {

    while (true) {
        std::unique_lock lock(local_mutex);
        cond_var.wait(lock, [this]() {
            return stop || num_points > 0;
        });
        if (stop)
            break;

        //Then we could get some work and do stuff
    }

}

void ThreadBase::start_thread() {
    native_thread = std::thread(&ThreadBase::idleLoop,this);
}


#endif //CHECKERENGINEX_THREAD_H
