//
// Created by robin on 8/1/18.
//


#include "Trainer.h"


int Trainer::getEpochs() {
    return epochs;
}

void Trainer::setEpochs(int epoch) {
    this->epochs = epoch;
}

void Trainer::setCValue(double cval) {
    this->cValue = cval;
}

double Trainer::getCValue() {
    return this->cValue;
}

double Trainer::getL2Reg() {
    return l2Reg;
}

double Trainer::getLearningRate() {
    return learningRate;
}

void Trainer::setl2Reg(double reg) {
    l2Reg = reg;
}

void Trainer::setLearningRate(double learn) {
    learningRate = learn;
}

double sigmoid(double c, double value) {
    return 1.0 / (1.0 + std::exp(c * value));
}

double sigmoidDiff(double c, double value) {
    return c * (sigmoid(c, value) * (sigmoid(c, value) - 1.0));
}

void Trainer::gradientUpdate(const Sample &sample) {
    //one step of stochastic gradient descent
    //one step of stochastic gradient descent
    Position x = sample.position;

    if (x.hasJumps(x.getColor()) || x.hasJumps(~x.getColor()))
        return;
    auto qStatic = gameWeights.evaluate<double>(x, 0);

    if (isWin((int) qStatic) || !isEval((int) qStatic))
        return;

    int num_wp = __builtin_popcount(x.WP & (~x.K));
    int num_bp = __builtin_popcount(x.BP & (~x.K));
    int num_wk = __builtin_popcount(x.WP & (x.K));
    int num_bk = __builtin_popcount(x.BP & (x.K));
    double phase = num_bp + num_wp + num_bk + num_wk;
    phase /= (double) (stage_size);
    double end_phase = stage_size - phase;
    end_phase /= (double) stage_size;


    double result;
    if (sample.result == -1)
        result = 0.0;
    else if (sample.result == 1)
        result = 1.0;
    else
        result = 0.5;
    double c = getCValue();

    double error = sigmoid(c, qStatic) - result;
    double color = x.color;
    const double temp = error * sigmoidDiff(c, qStatic);;
    accu_loss += error * error;
    auto x_flipped = (x.getColor() == BLACK) ? x.getColorFlip() : x;

    const size_t sub_offset =0;
    for (size_t p = 0; p < 2; ++p) {
        double diff = temp;
        if (p == 0) {
            //derivative for the opening
            diff *= phase * color;
        } else {
            //derivative for the ending
            diff *= end_phase * color;
        }

        for (auto i = 0; i < 3; ++i) {
            for (auto k = 0; k < 3; ++k) {
                const uint32_t sub_reg = region << (8 * i + k);
                size_t index = getIndex2(sub_reg, x_flipped);
                size_t sub_index = 18 * index + 2 * k + 6 * i + p + sub_offset;
                momentums[sub_index] = beta * momentums[sub_index] + (1.0 - beta) * diff;
                gameWeights[sub_index] = gameWeights[sub_index] - getLearningRate() * momentums[sub_index];
            }

        }
    }

    //for king_op
    {
        double diff = temp;
        double d = phase * ((double) (num_wk - num_bk));
        diff *= d;
        momentums[SIZE] = beta * momentums[SIZE] + (1.0 - beta) * diff;
        gameWeights.kingOp = gameWeights.kingOp - getLearningRate() * momentums[SIZE];
    }

    //for king_end
    {
        double diff = temp;
        double d = end_phase * ((double) (num_wk - num_bk));
        diff *= d;
        momentums[SIZE + 1] = beta * momentums[SIZE + 1] + (1.0 - beta) * diff;
        gameWeights.kingEnd = gameWeights.kingEnd - getLearningRate() * momentums[SIZE + 1];
    }

    {
        //for tempo ranks black side
        uint32_t man = x.BP & (~x.K);
        for (size_t i = 0; i < gameWeights.tempo_ranks.size(); ++i) {
            double diff = temp;
            diff *= -1.0;
            uint32_t shift = 4 * i;
            uint32_t index = man >> shift;
            index &= temp_mask;
            const size_t mom_index = SIZE + 2 + i * 16 + index;
            momentums[mom_index] = beta * momentums[mom_index] + (1.0 - beta) * diff;
            gameWeights.tempo_ranks[i][index] =
                    gameWeights.tempo_ranks[i][index] - getLearningRate() * momentums[mom_index];
        }
        //for tempo ranks white-side
        man = x.WP & (~x.K);
        man = getMirrored(man);
        for (size_t i = 0; i < gameWeights.tempo_ranks.size(); ++i) {
            double diff = temp;
            diff *= 1.0;
            uint32_t shift = 4 * i;
            uint32_t index = man >> shift;
            index &= temp_mask;
            const size_t mom_index = SIZE + 2 + i * 16 + index;
            momentums[mom_index] = beta * momentums[mom_index] + (1.0 - beta) * diff;
            gameWeights.tempo_ranks[i][index] =
                    gameWeights.tempo_ranks[i][index] - getLearningRate() * momentums[mom_index];
        }

    }

}


void Trainer::epoch() {
    static std::mt19937_64 generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::cout << "Start shuffling" << std::endl;
    std::shuffle(data.begin(), data.end(), generator);
    std::cout << "Done shuffling" << std::endl;
    int counter = 0;
    std::for_each(data.begin(), data.end(),
                  [this, &counter](Sample sample) {
                      auto num_pieces = Bits::pop_count(sample.position.BP | sample.position.WP);
                      uint32_t WP = sample.position.WP & (~sample.position.K);
                      uint32_t BP = sample.position.BP & (~sample.position.K);
                      uint32_t WK = sample.position.WP & (sample.position.K);
                      uint32_t BK = sample.position.BP & (sample.position.K);

                      if (num_pieces > 24 || std::abs(sample.position.color) != 1 || num_pieces == 0 ||
                          ((WP & BP) != 0) || ((WK & BK) != 0)) {
                      } else {
                          counter++;
                          gradientUpdate(sample);
                      }

                  });

}


void Trainer::startTune() {
    int counter = 0;
    std::cout << "Data_size: " << data.size() << std::endl;
    while (counter < getEpochs()) {
        std::cout << "Start of epoch: " << counter << "\n\n" << std::endl;
        std::cout << "CValue: " << getCValue() << std::endl;
        double num_games = data.size();
        double loss = accu_loss / ((double) num_games);
        loss = sqrt(loss);
        accu_loss = 0.0;
        last_loss_value = loss;
        std::cout << "Loss: " << loss << std::endl;
        auto t1 = std::chrono::high_resolution_clock::now();
        epoch();
        auto t2 = std::chrono::high_resolution_clock::now();
        auto dur = t2-t1;
        std::cout<<"Time for epoch: "<<dur.count()/1000000<<std::endl;
        counter++;
        std::string name = "X" + std::to_string(counter) + ".weights";
        gameWeights.storeWeights(name);
        gameWeights.storeWeights("bloom.weights");
        std::cout << "LearningRate: " << learningRate << std::endl;
        std::cout << "NonZero: " << gameWeights.numNonZeroValues() << std::endl;
        std::cout << "Max: " << gameWeights.getMaxValue() << std::endl;
        std::cout << "Min: " << gameWeights.getMinValue() << std::endl;
        std::cout << "kingScore:" << gameWeights.kingOp << " | " << gameWeights.kingEnd << std::endl;
        learningRate = learningRate * (1.0 - decay);
    }
}

double Trainer::calculateLoss() {
    //I dont get it

    auto evalLambda = [this](const Sample sample) {
        if (sample.result == 1000)
            return 0.0;
        Position pos = sample.position;
        Board board;
        board.getPosition().BP = pos.BP;
        board.getPosition().WP = pos.WP;
        board.getPosition().K = pos.K;
        board.getPosition().color = pos.color;

        double result;
        if (sample.result == -1)
            result = 0.0;
        else if (sample.result == 1)
            result = 1.0;
        else
            result = 0.5;
        auto color = static_cast<double>(board.getPosition().getColor());
        Line local;
        double quiesc = color * searchValue(board, 0, 10000, false);
        double current = result - sigmoid(cValue, quiesc);
        current = current * current;
        return current;
    };
    double result = 0.0;
    std::for_each(data.begin(), data.end(), [&](const Sample &sample) {
        result += evalLambda(sample);
    });

    return std::sqrt(result / static_cast<double>(data.size()));
}

