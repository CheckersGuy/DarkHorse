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
#include <numeric>
#include <execution>
#include "Utilities.h"


#ifdef TRAIN
extern Weights<double> gameWeights;
#endif


using namespace Training;
class Trainer {

private:
    int epochs;
    uint64_t bufferCounter;
    std::vector<TrainingPos> &data;
    double learningRate, l2Reg, cValue;

public:


    Trainer(std::vector<TrainingPos> &data) :data(data), cValue(1.0),
                                                                      learningRate(0.1), l2Reg(0.01),bufferCounter(0){};
    void epoch();

    void gradientUpdate(TrainingPos position);

    void setEpochs(int epoch);

    void setCValue(double cval);

    void setl2Reg(double reg);

    void setLearningRate(double learn);

    double getCValue();

    int getEpochs();

    double getLearningRate();

    double getL2Reg();

    void startTune();

    double calculateLoss();

    static double evaluatePosition( Board&board,Weights<double>& weights,size_t index,double offset);

};


#endif //TRAINING_TRAINER_H
