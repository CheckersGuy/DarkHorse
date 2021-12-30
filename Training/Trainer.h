//
// Created by robin on 8/1/18.
//

#ifndef TRAINING_TRAINER_H
#define TRAINING_TRAINER_H

#include <condition_variable>
#include "Weights.h"
#include "queue"
#include <numeric>
#include <execution>
#include "Utilities.h"
#include "Generator.h"
#include <types.h>
#include <PosStreamer.h>

class Trainer {

private:
    int epochs;
    double learningRate, l2Reg, cValue;
    double accu_loss{0};
    double last_loss_value;
    double beta{0.99};
    double decay{0.07};
    std::unique_ptr<double[]> momentums;
    std::mt19937_64 generator;
    PosStreamer pos_streamer;
    std::string weights_path;
    size_t step_counter{0};
    size_t epoch_counter{0};
    size_t save_point_step{1000000};
public:


    Trainer(const std::string &data_path) : cValue(1.0),
                                            learningRate(0.1), last_loss_value(std::numeric_limits<double>::max()),
                                            l2Reg(0.05), generator(std::mt19937_64(231231241ull)),
                                            pos_streamer(PosStreamer(data_path, 5000000)) {
        momentums = std::make_unique<double[]>(SIZE + 2u + 16u * 7u+4);

    };

    void epoch();


    void gradientUpdate(Sample &sample);

    void setEpochs(int epoch);

    void set_weights_path(std::string path);

    void set_savepoint_step(size_t num_steps);

    void set_decay(double d);

    void setCValue(double cval);

    void setl2Reg(double reg);

    void setLearningRate(double learn);

    double getCValue();

    int getEpochs();

    double getLearningRate();

    void startTune();

};


#endif //TRAINING_TRAINER_H
