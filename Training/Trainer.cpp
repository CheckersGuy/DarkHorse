//
// Created by robin on 8/1/18.
//


#include "Trainer.h"


int Trainer::get_num_epochs() {
    return epochs;
}

void Trainer::set_epochs(int epoch) {
    this->epochs = epoch;
}

void Trainer::set_c_value(double cval) {
    this->cValue = cval;
}

double Trainer::get_c_value() {
    return this->cValue;
}


double Trainer::get_learning_rate() {
    return learningRate;
}

void Trainer::set_learning_rate(double learn) {
    learningRate = learn;
}

double sigmoid(double c, double value) {
    return 1.0 / (1.0 + std::exp(c * value));
}

double sigmoidDiff(double c, double value) {
    return c * (sigmoid(c, value) * (sigmoid(c, value) - 1.0));
}

void Trainer::gradient_update(Sample &sample) {
    Position x = sample.position;
    if (sample.result == UNKNOWN)
        return;

    if (x.has_jumps(x.get_color()) || x.has_jumps(~x.get_color()))
        return;
    double qStatic = weights.evaluate<double>(x, 0);

    if (isWin((int) qStatic))
        return;

    step_counter++;
    if (!weights_path.empty()) {
        if ((step_counter % save_point_step) == 0) {
            weights.store_weights(weights_path);
            //saving the trainer state
        }
    } else {
        std::cerr << "Weights path was empty" << std::endl;
        std::exit(-1);
    }

    auto adam_update = [&](size_t param, double gradient, double weight) {
        m[param] = beta_one * m[param] + (1.0 - beta_one) * gradient;
        v[param] = beta_two * v[param] + (1.0 - beta_two) * gradient * gradient;
        auto m_hat = m[param] * (1.0 - beta_one_t[param]);
        auto v_hat = v[param] * (1.0 - beta_two_t[param]);

        double alpha = get_learning_rate();
        beta_one_t[param] *= beta_one;
        beta_two_t[param] *= beta_one;

        return (alpha * m_hat / ((std::sqrt(v_hat) + 0.000001)));
       //return alpha * m[param];

    };


    int num_wp = Bits::pop_count(x.WP & (~x.K));
    int num_bp = Bits::pop_count(x.BP & (~x.K));
    int num_wk = Bits::pop_count(x.WP & (x.K));
    int num_bk = Bits::pop_count(x.BP & (x.K));
    double tot_pieces = num_bp + num_wp + num_bk + num_wk;
    double phase = tot_pieces / ((double)stage_size);
    double end_phase = 1.0-phase;


    double result;
    if (sample.result == BLACK_WON)
        result = 0.0;
    else if (sample.result == WHITE_WON)
        result = 1.0;
    else
        result = 0.5;
    double c = get_c_value();

    double error = sigmoid(c, qStatic) - result;
    double color = x.color;
    const double temp = error * sigmoidDiff(c, qStatic);;
    accu_loss += error * error;

    auto x_flipped = (x.get_color() == BLACK) ? x.get_color_flip() : x;


    for (size_t p = 0; p < 2; ++p) {
        double diff = temp;
        if (p == 0) {
            //derivative for the opening
            diff *= phase * color;
        } else {
            //derivative for the ending
            diff *= end_phase * color;
        }

        auto f = [&](size_t index) {
            size_t sub_index = index + p;
            weights[sub_index] = weights[sub_index] - adam_update(sub_index, diff, weights[sub_index]);
        };

        if (x_flipped.K == 0) {
            Bits::big_index(f, x_flipped.WP, x_flipped.BP, x_flipped.K);
        } else {
            Bits::small_index(f, x_flipped.WP, x_flipped.BP, x_flipped.K);
        }
    }

    //for king_op
    {
        double diff = temp;
        double d = phase * ((double) (num_wk - num_bk));
        diff *= d;
        weights.kingOp = weights.kingOp - adam_update(SIZE, diff, weights.kingOp);
    }

    //for king_end
    {
        double diff = temp;
        double d = end_phase * ((double) (num_wk - num_bk));
        diff *= d;
        weights.kingEnd = weights.kingEnd - adam_update(SIZE + 1, diff, weights.kingEnd);
    }

    {
        //for tempo ranks black side
        uint32_t man = x.BP & (~x.K);
        for (size_t i = 0; i < 7; ++i) {
            double diff = temp;
            diff *= -1.0;
            uint32_t shift = 4 * i;
            uint32_t index = man >> shift;
            index &= temp_mask;
            const size_t mom_index = SIZE + 2 + i * 16 + index;
            weights.tempo_ranks[i][index] =
                    weights.tempo_ranks[i][index] - adam_update(mom_index, diff, weights.tempo_ranks[i][index]);
        }
        //for tempo ranks white-side
        man = x.WP & (~x.K);
        man = getMirrored(man);
        for (size_t i = 0; i < 7; ++i) {
            double diff = temp;
            diff *= 1.0;
            uint32_t shift = 4 * i;
            uint32_t index = man >> shift;
            index &= temp_mask;
            const size_t mom_index = SIZE + 2 + i * 16 + index;

            weights.tempo_ranks[i][index] =
                    weights.tempo_ranks[i][index] - adam_update(mom_index, diff, weights.tempo_ranks[i][index]);
        }

    }

}

void Trainer::set_weights_path(std::string path) {
    this->weights_path = path;
}

void Trainer::set_savepoint_step(size_t num_steps) {
    //we save the weights every num_steps
    save_point_step = num_steps;
}

void Trainer::set_weight_decay(double d_value) {
    l2Reg = d_value;
}

void Trainer::set_decay(double d) {
    decay = d;
}

void Trainer::epoch() {
    size_t num_samples = pos_streamer.get_num_positions();
    for (auto i = size_t{0}; i < num_samples; ++i) {
        Sample sample = pos_streamer.get_next();
        if (!sample.position.is_legal()) {
            continue;
        }

        gradient_update(sample);
    }


    epoch_counter++;

}


void Trainer::start_tune() {
    //


    int counter = 0;
    std::cout <<"data_size: " << pos_streamer.get_num_positions() << "\n";
    while (counter < get_num_epochs()) {
        std::stringstream ss_stream;
        ss_stream.clear();
        ss_stream<<std::setfill('-')<<std::setw(40)<<"\n";
        ss_stream << "Start of epoch: " << counter << "\n" <<"\n";
        ss_stream<<std::setfill('-')<<std::setw(40)<<"\n";
        ss_stream << "CValue: " << get_c_value() << "\n";
        double num_games = (double) (pos_streamer.get_num_positions());
        double loss = accu_loss / ((double) num_games);
        loss = sqrt(loss);
        accu_loss = 0.0;
        last_loss_value = loss;
        ss_stream<<std::setfill('-')<<std::setw(40)<<"\n";
        ss_stream << "Loss: " << loss << "\n";
        auto t1 = std::chrono::high_resolution_clock::now();
        epoch();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto dur = t2 - t1;
        ss_stream<<std::setfill('-')<<std::setw(40)<<"\n";
        ss_stream << "Time for epoch: " << dur.count() / 1000000 <<"\n";
        counter++;
        ss_stream<<std::setfill('-')<<std::setw(40)<<"\n";
        ss_stream << "LearningRate: " << learningRate <<"\n";
        ss_stream<<std::setfill('-')<<std::setw(40)<<"\n";
        ss_stream  << "NonZero: " << weights.num_non_zero_weights() << "\n";
        ss_stream<<std::setfill('-')<<std::setw(40)<<"\n";
        ss_stream  << "Max: " << weights.get_max_weight() << "\n";
        ss_stream<<std::setfill('-')<<std::setw(40)<<"\n";
        ss_stream  << "Min: " << weights.get_min_weight() << "\n";
        ss_stream<<std::setfill('-')<<std::setw(40)<<"\n";
        ss_stream  << "kingScore:" << weights.kingOp << " | " << weights.kingEnd <<"\n";
        learningRate = learningRate * (1.0 - decay);
        std::cout<<ss_stream.str()<<std::endl;
    }

}


