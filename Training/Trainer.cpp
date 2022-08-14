//
// Created by robin on 8/1/18.
//


#include "Trainer.h"

double sigmoid(double value) {
    if(value>=0){
        double z =std::exp(-value);
        return 1.0/(1.0+z);
    }else{
        double z =std::exp(value);
        return z/(1.0+z);
    }
}

double sigmoid(double c_value,double value){
    return sigmoid(c_value*value);
}
double sigmoidDiff(double c, double value) {
    return c*(sigmoid(c,value) * (1.0-sigmoid(c,value) ));
}

void Trainer::set_train_file_locat(std::string train_f){
    train_file = train_f;
}

void Trainer::save_trainer_state(std::string output_file) {
    std::ofstream stream(output_file, std::ios::binary);
    if(!stream.good()){
        std::cerr<<"Could not save trainer state"<<std::endl;
        return;
    }
    weights.store_weights(stream);
    stream.write((char*)&epoch_counter,sizeof(epoch_counter));
    stream.write((char*)&step_counter,sizeof(step_counter));
    stream.write((char *) m.get(), sizeof(double) * num_weights);
    stream.write((char *) v.get(), sizeof(double) * num_weights);
    stream.write((char *) beta_one_t.get(), sizeof(double) * num_weights);
    stream.write((char *) beta_two_t.get(), sizeof(double) * num_weights);
    stream.write((char *) momentums.get(), sizeof(double) * num_weights);
    stream.write((char *) &beta_one, sizeof(double));
    stream.write((char *) &beta_two, sizeof(double));
    stream.write((char *) &decay, sizeof(double));
    stream.write((char *) &learningRate, sizeof(double));
    stream.write((char*)&cValue,sizeof(double));
}

void Trainer::load_trainer_state(std::string input_file) {
    std::ifstream stream(input_file, std::ios::binary);
       if(!stream.good()){
        std::cerr<<"Could not load trainer state"<<std::endl;
        return;
    }
    weights.load_weights(stream);
    stream.read((char*)&epoch_counter,sizeof(epoch_counter));
    stream.read((char*)&step_counter,sizeof(step_counter));
    stream.read((char *) m.get(), sizeof(double) * num_weights);
    stream.read((char *) v.get(), sizeof(double) * num_weights);
    stream.read((char *) beta_one_t.get(), sizeof(double) * num_weights);
    stream.read((char *) beta_two_t.get(), sizeof(double) * num_weights);
    stream.read((char *) momentums.get(), sizeof(double) * num_weights);
    stream.read((char *) &beta_one, sizeof(double));
    stream.read((char *) &beta_two, sizeof(double));
    stream.read((char *) &decay, sizeof(double));
    stream.read((char *) &learningRate, sizeof(double));
    stream.read((char*)&cValue,sizeof(double));
}


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

void Trainer::gradient_update(Sample &sample) {
    Position x = sample.position;

    double qStatic = weights.evaluate<double>(x, 0);

    if (isWin((int) qStatic))
        return;

    step_counter++;
    if (!weights_path.empty()) {
        if ((step_counter % save_point_step) == 0) {
            weights.store_weights(weights_path);
            std::cout << "Saved weights" << std::endl;
            //saving the trainer state
        }
    } else {
        std::cerr << "Weights path was empty" << std::endl;
        std::exit(-1);
    }

    auto adam_update = [&](size_t param, double gradient, double weight) {
        double alpha = get_learning_rate();
        m[param] = beta_one * m[param] + (1.0 - beta_one) * gradient;
         v[param] = beta_two * v[param] + (1.0 - beta_two) * gradient * gradient;
        auto m_hat = m[param] * (1.0 - beta_one_t[param]);
        auto v_hat = v[param] * (1.0 - beta_two_t[param]);

        
        beta_one_t[param] *= beta_one;
        beta_two_t[param] *= beta_one;

        //return (alpha * m_hat / ((std::sqrt(v_hat) + 0.000001)))+alpha*l2Reg*weights[param];  
        return alpha * m[param];

    };


    const double stage_size = 24.0;

    double num_wp = Bits::pop_count(x.WP & (~x.K));
    double num_bp = Bits::pop_count(x.BP & (~x.K));
    double num_wk = Bits::pop_count(x.WP & (x.K));
    double num_bk = Bits::pop_count(x.BP & (x.K));
    double tot_pieces = num_bp + num_wp + num_bk + num_wk;
    double phase = tot_pieces /stage_size;
    double end_phase = 1.0 - phase;


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
        double d = phase * ((num_wk - num_bk));
        diff *= d;
        weights.kingOp = weights.kingOp - adam_update(SIZE, diff, weights.kingOp);
    }

    //for king_end
    {
        double diff = temp;
        double d = end_phase * ((num_wk - num_bk));
        diff *= d;
        weights.kingEnd = weights.kingEnd - adam_update(SIZE + 1, diff, weights.kingEnd);
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
        Sample sample;
         do{
            sample = pos_streamer.get_next();
         }while(sample.result == UNKNOWN || (sample.position.has_jumps()));

        gradient_update(sample);
    }
     //saving the trainer state

    epoch_counter++;

}


void Trainer::start_tune() {
    //


    std::cout << "data_size: " << pos_streamer.get_num_positions() << "\n";
    while (epoch_counter < get_num_epochs()) {

        std::stringstream ss_stream;
        ss_stream.clear();
        ss_stream << std::setfill('-') << std::setw(40) << "\n";
        ss_stream << "Start of epoch: " << epoch_counter << "\n" << "\n";
        ss_stream << std::setfill('-') << std::setw(40) << "\n";
        ss_stream << "CValue: " << get_c_value() << "\n";
        double num_games = (double) (pos_streamer.get_num_positions());
        double loss = accu_loss / ((double) num_games);
        loss = sqrt(loss);
        accu_loss = 0.0;
        last_loss_value = loss;
        ss_stream << std::setfill('-') << std::setw(40) << "\n";
        ss_stream << "Loss: " << loss << "\n";

        ss_stream << std::setfill('-') << std::setw(40) << "\n";
        ss_stream << "LearningRate: " << learningRate << "\n";
        ss_stream << std::setfill('-') << std::setw(40) << "\n";
        ss_stream << "NonZero: " << weights.num_non_zero_weights() << "\n";
        ss_stream << std::setfill('-') << std::setw(40) << "\n";
        ss_stream << "Max: " << weights.get_max_weight() << "\n";
        ss_stream << std::setfill('-') << std::setw(40) << "\n";
        ss_stream << "Min: " << weights.get_min_weight() << "\n";
        ss_stream << std::setfill('-') << std::setw(40) << "\n";
        ss_stream << "kingScore:" << weights.kingOp << " | " << weights.kingEnd << "\n";
        learningRate = learningRate * (1.0 - decay);
        std::cout << ss_stream.str() << std::endl;

        auto t1 = std::chrono::high_resolution_clock::now();
        epoch();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto dur = t2 - t1;
        ss_stream << std::setfill('-') << std::setw(40) << "\n";
        std::cout<< "Time for epoch: " << dur.count() / 1000000 << "\n";
        save_trainer_state(train_file);
    }
    weights.store_weights(weights_path);
   
}


