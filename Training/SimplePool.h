//
// Created by robin on 6/21/18.
//

#ifndef TRAINING_SIMPLEPOOL_H
#define TRAINING_SIMPLEPOOL_H

#include<atomic>
#include <thread>
#include <mutex>
#include <vector>
#include <queue>
#include <iostream>
#include <SMPLock.h>
#include "Training.h"


bool simGame(Training::TrainingGame &one, Training::TrainingGame &two, float threshHold);


class SimplePool {


private:
    float threshHold;
    std::vector<Training::TrainingGame> &games;
    std::vector<Training::TrainingGame> &removed;
    std::vector<std::thread> workThreads;
    SMPLock myMutex;
    std::atomic<int> workCounter;

public:


    SimplePool(int threads, std::vector<Training::TrainingGame> &games, std::vector<Training::TrainingGame> &removed,
               float threshHold);

    SimplePool(int threads, std::vector<Training::TrainingGame> &games, std::vector<Training::TrainingGame> &removed);

    ~SimplePool();

    void init();

    void joinAll();

    void waitAll();
};


#endif //TRAINING_SIMPLEPOOL_H
