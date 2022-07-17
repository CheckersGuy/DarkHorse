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
#include <sstream>
class Trainer {

private:
    constexpr static size_t num_weights =SIZE + 2u + 16u * 7u + 4;
    size_t epochs{0};
    double learningRate, l2Reg, cValue;
    double accu_loss{0};
    double last_loss_value;
    double beta{0.99};
    double beta_one{0.9}, beta_two{0.999};
    double decay{0.07};
    std::unique_ptr<double[]> momentums;
    std::unique_ptr<double[]> m;
    std::unique_ptr<double[]> v;
    std::unique_ptr<double[]> beta_one_t;
    std::unique_ptr<double[]> beta_two_t;

    std::mt19937_64 generator;
    PosStreamer pos_streamer;
    std::string weights_path;
    std::string train_file; //saves the entire state of the trainer to the location
    size_t step_counter{0};
    size_t epoch_counter{0};
    size_t save_point_step{1000000};
    Weights<double> weights;
    //weights for the lookahead optimizer
public:


    Trainer(const std::string &data_path) : cValue(1.0),
                                            learningRate(0.1), last_loss_value(std::numeric_limits<double>::max()),
                                            l2Reg(0.05), generator(std::mt19937_64(231231241ull)),
                                            pos_streamer(PosStreamer(data_path, 10000000)) {
        const size_t MAX_SIZE = SIZE + (2u + 16u * 7u + 4);
        momentums = std::make_unique<double[]>(MAX_SIZE);
        pos_streamer.set_input_format(InputFormat::PATTERN);
        m = std::make_unique<double[]>(MAX_SIZE);
        v = std::make_unique<double[]>(MAX_SIZE);
        beta_one_t = std::make_unique<double[]>(MAX_SIZE);
        beta_two_t = std::make_unique<double[]>(MAX_SIZE);

        for (auto i = 0; i < MAX_SIZE; ++i) {
            m[i] = 0;
            v[i] = 0;
            beta_one_t[i] = beta_one;
            beta_two_t[i] = beta_two;
        }

    };

    void save_trainer_state(std::string output_file);

    void load_trainer_state(std::string input_file);

    void epoch();

    void gradient_update(Sample &sample);

    void set_epochs(int epoch);

    void set_weights_path(std::string path);

    void set_savepoint_step(size_t num_steps);

    void set_train_file_locat(std::string train_file);

    void set_weight_decay(double d_value);

    void set_decay(double d);

    void set_c_value(double cval);

    void set_learning_rate(double learn);

    double get_c_value();

    int get_num_epochs();

    double get_learning_rate();

    void start_tune();


};

double sigmoid(double value);

double sigmoid(double c_value,double value);
double sigmoidDiff(double c, double value);
#endif //TRAINING_TRAINER_H
