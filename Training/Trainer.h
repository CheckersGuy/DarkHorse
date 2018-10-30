//
// Created by robin on 8/1/18.
//

#ifndef TRAINING_TRAINER_H
#define TRAINING_TRAINER_H

#include <condition_variable>
#include <atomic>
#include "Weights.h"
#include "Training.h"
#include "queue"

struct WorkerPool {
    Value *results;
    Weights<double> *weights;
    int threads, batchSize, gameCounter;
    std::vector<std::thread> workers;
    std::queue<Position> work;
    std::mutex myMutex;

    WorkerPool(int batchSize, int threads, Weights<double> &weights) : batchSize(batchSize), threads(threads),
                                                                       weights(&weights), gameCounter(0) {
    }

    ~WorkerPool() {
        for (auto &th : workers) {
            th.join();
        }
    }

    static void idleLoop(WorkerPool *pool);

    void waitAll();

    void setOutput(Value *val);

    void startThreads();

    void addWork(Position pos);

    Value operator[](int index);
};


class Trainer {

private:
    int epochs;
    int threads;
    Weights<double> &weights;
    Training::TrainingData &data;
    double learningRate, l2Reg, cValue;


public:


    Trainer(Weights<double> &weights, Training::TrainingData &data) : weights(weights), data(data), cValue(1.0),
                                                                      learningRate(0.1), l2Reg(0.01) {}

    void epoch();

    void setThreads(int threads);

    void setEpochs(int epoch);

    void setCValue(double cval);

    void setl2Reg(double reg);

    void setLearningRate(double learn);

    double getCValue();

    int getThreads();

    int getEpochs();

    double getLearningRate();

    double getL2Reg();

    void startTune();

    double calculateLoss();

};


#endif //TRAINING_TRAINER_H
