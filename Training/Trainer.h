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
#include "Generator.h"

class Trainer {

private:
    int epochs;
    std::vector<Sample> data;
    double learningRate, l2Reg, cValue;
    double accu_loss{0};
    double last_loss_value;
    double beta{0.99};
    double decay{0.001};
    std::unique_ptr<double[]> momentums;

public:


    Trainer(const std::string &data_path) : cValue(1.0),
                                            learningRate(0.1), last_loss_value(std::numeric_limits<double>::max()),
                                            l2Reg(0.01) {
        momentums = std::make_unique<double[]>(SIZE + 2u + 16u * 7u);
        Utilities::read_binary<Sample>(std::back_inserter(data), data_path);
        std::cout<<"Number of samples "<<data.size()<<std::endl;
    };

    void epoch();


    void gradientUpdate(const Sample &sample);

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

};


#endif //TRAINING_TRAINER_H
