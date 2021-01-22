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


double getWinValue(Training::Result result) {
    if (result == Training::BLACK_WON)
        return 0.0;
    else if (result == Training::WHITE_WON)
        return 1.0;

    return 0.5;

}

double sigmoid(double c, double value) {
    return 1.0 / (1.0 + std::exp(c * value));
}

double sigmoidDiff(double c, double value) {
    return c * (sigmoid(c, value) * (sigmoid(c, value) - 1.0));
}


void Trainer::gradientUpdate(const Training::Position &pos) {
    //one step of stochastic gradient descent
    //one step of stochastic gradient descent
    Position x;
    x.BP = pos.bp();
    x.WP = pos.wp();
    x.K = pos.k();
    x.color = (pos.mover() == Training::BLACK) ? BLACK : WHITE;

    if (pos.result() == Training::UNKNOWN)
        return;

    if (x.hasJumps(x.getColor()))
        return;
    auto qStatic = gameWeights.evaluate<double>(x, 0);


    int num_wp = __builtin_popcount(x.WP & (~x.K));
    int num_bp = __builtin_popcount(x.BP & (~x.K));
    int num_wk = __builtin_popcount(x.WP & (x.K));
    int num_bk = __builtin_popcount(x.BP & (x.K));
    double phase = num_bp + num_wp + num_bk + num_wk;
    phase /= (double) (stage_size);
    double end_phase = stage_size - phase;
    end_phase /= (double) stage_size;


    if (isWin(qStatic) || !isEval(qStatic))
        return;

    double result = getWinValue(pos.result());
    double c = getCValue();

    auto qValue = qStatic;
    double error = sigmoid(c, qValue) - result;
    const double temp = error * sigmoidDiff(c, qValue);;
    accu_loss += error * error;
    for (size_t p = 0; p < 2; ++p) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8u * j + i);
                size_t index = 18ull * getIndex2(curRegion, x) + 9ull * p + 3ull * j + i;
                double diff = temp;
                if (p == 0) {
                    //derivative for the openi
                    diff *= phase;
                } else {
                    //derivative for the oending
                    diff *= end_phase;
                }
                momentums[index] = beta * momentums[index] + (1.0 - beta) * diff;
                //need to update the momentum term
                gameWeights[index] = gameWeights[index] - getLearningRate() * momentums[index];

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
    std::shuffle(data.mutable_positions()->begin(), data.mutable_positions()->end(), generator);
    std::cout << "Done shuffling" << std::endl;
    int counter = 0;


    std::for_each(data.mutable_positions()->begin(), data.mutable_positions()->end(),
                  [this, &counter](Training::Position &pos) {
                      counter++;
                      gradientUpdate(pos);
                  });
}


void Trainer::startTune() {
    int counter = 0;
    std::cout << "Data_size: " << data.positions_size() << std::endl;
    while (counter < getEpochs()) {
        std::cout << "Start of epoch: " << counter << "\n\n" << std::endl;
        std::cout << "CValue: " << getCValue() << std::endl;
        double num_games = data.positions().size();
        double loss = accu_loss / ((double) num_games);
        loss = sqrt(loss);
        accu_loss = 0.0;
        last_loss_value = loss;
        std::cout << "Loss: " << loss << std::endl;
        epoch();
        counter++;
        std::string name = "X" + std::to_string(counter) + ".weights";
        gameWeights.storeWeights(name);
        gameWeights.storeWeights("current.weights");
        std::cout<<"LearningRate: "<<learningRate<<std::endl;
        std::cout << "NonZero: " << gameWeights.numNonZeroValues() << std::endl;
        std::cout << "Max: " << gameWeights.getMaxValue() << std::endl;
        std::cout << "Min: " << gameWeights.getMinValue() << std::endl;
        std::cout << "kingScore:" << gameWeights.kingOp << " | " << gameWeights.kingEnd << std::endl;
        std::cout << "TEMPO_RANKS" << std::endl;
        std::cout << "TEMPO_RANKS" << std::endl;
        for (auto i = 0; i < gameWeights.tempo_ranks.size(); ++i) {
            std::copy(gameWeights.tempo_ranks[i].begin(),
                      gameWeights.tempo_ranks[i].end(),
                      std::ostream_iterator<double>(std::cout, ","));
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout << std::endl;

        learningRate = (1.0 - decay) * learningRate;
    }
}

double Trainer::calculateLoss() {
    //I dont get it

    auto evalLambda = [this](const Training::Position &pos) {
        if (pos.result() == Training::UNKNOWN)
            return 0.0;
        Board board;
        board.getPosition().BP = pos.bp();
        board.getPosition().WP = pos.wp();
        board.getPosition().K = pos.k();
        board.getPosition().color = ((pos.mover() == Training::BLACK) ? BLACK : WHITE);


        double result = getWinValue(pos.result());
        auto color = static_cast<double>(board.getPosition().getColor());
        Line local;
        double quiesc = color * searchValue(board, 0, 10000, false);
        double current = result - sigmoid(cValue, quiesc);
        current = current * current;
        return current;
    };
    double result = 0.0;
    std::for_each(data.mutable_positions()->begin(), data.mutable_positions()->end(), [&](Training::Position &pos) {
        result += evalLambda(pos);
    });

    return std::sqrt(result / static_cast<double>(data.positions_size()));
}

