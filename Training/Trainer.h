//
// Created by robin on 8/1/18.
//

#ifndef TRAINING_TRAINER_H
#define TRAINING_TRAINER_H

#include <condition_variable>
#include <SMPLock.h>
#include "Weights.h"
#include "queue"
#include <numeric>
#include <execution>
#include "Utilities.h"
#include "proto/Training.pb.h"

#ifdef TRAIN
extern Weights<double> gameWeights;
#endif


class Trainer {

private:
    int epochs;
    Training::TrainData data;
    double learningRate, l2Reg, cValue;
    double last_loss_value;

public:


    Trainer(const std::string& data_path) : cValue(1.0),
                                                                      learningRate(0.1),last_loss_value(std::numeric_limits<double>::max()), l2Reg(0.01){
        std::ifstream stream(data_path);
        if(!stream.good()){
            std::cerr<<"Couldnt init training data"<<std::endl;
            std::exit(EXIT_FAILURE);
        }
        data.ParseFromIstream(&stream);
        stream.close();
    };
    void epoch();


    void gradientUpdate(Training::Position& position);

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
