//
// Created by robin on 6/21/18.
//

#include "SimplePool.h"


bool simGame(Training::TrainingGame &one, Training::TrainingGame &two, float threshHold) {
    if (one.result != two.result) {
        return false;
    }

    float counter = 0;
    float size = std::min(one.positions.size(), two.positions.size());

    for (int i = 0; i < size; ++i) {
        if (one.positions[i] == two.positions[i]) {
            counter++;
        }
        if ((counter / size) >= threshHold) {
            return true;
        }

    }
    return false;
}


std::mutex myMutex;


void SimplePool::init() {

    while (workCounter < games.size()) {
        int index = workCounter.fetch_add(1);
        bool add = true;
        for (int k = index + 1; k < games.size(); ++k) {
            if (simGame(games[index], games[k], threshHold)) {
                add = false;
                break;
            }
        }
        myMutex.lock();
        if (add) {
            removed.emplace_back(games[index]);
        }
        myMutex.unlock();

    }
}

SimplePool::SimplePool(int threads, std::vector<Training::TrainingGame> &games,
                       std::vector<Training::TrainingGame> &removed, float threshHold) : workCounter(
        0), games(games), removed(removed), threshHold(threshHold) {
    for (int i = 0; i < threads; ++i) {
        std::cout << "Started the thread" << std::endl;
        workThreads.push_back(std::thread(&SimplePool::init, this));
    }
}

SimplePool::SimplePool(int threads, std::vector<Training::TrainingGame> &games,
                       std::vector<Training::TrainingGame> &removed) : workCounter(
        0), games(games), removed(removed), threshHold(0.9) {
    for (int i = 0; i < threads; ++i) {
        std::cout << "Started the thread" << std::endl;
        workThreads.push_back(std::thread(&SimplePool::init, this));
    }
}

SimplePool::~SimplePool() {
    for (std::thread &current : workThreads) {
        current.join();
    }
}


void SimplePool::joinAll() {
    for (std::thread &current : workThreads) {
        current.join();
    }
}


void SimplePool::waitAll() {
    //Waits until all of the work is finished
    while (workCounter < games.size());
}



