//
// Created by robin on 8/1/18.
//

#ifndef TRAINING_TRAINER_H
#define TRAINING_TRAINER_H

#include <condition_variable>
#include <atomic>
#include <SMPLock.h>
#include "Weights.h"
#include "Training.h"
#include "queue"



extern Weights<double> gameWeights;

class Trainer {

private:
    int epochs;
    int batchSize;
    int threads;
    Training::TrainingData &data;
    double learningRate, l2Reg, cValue;

public:


    Trainer(Weights<double> &weights, Training::TrainingData &data) :data(data), cValue(1.0),
                                                                      learningRate(0.1), l2Reg(0.01),threads(1) {


    }

    void epoch();

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
