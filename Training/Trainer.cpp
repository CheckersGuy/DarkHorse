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

    if (value > 0) {
        return 1.0 / (1.0 + std::exp(c * value));
    } else {
        return std::exp(-c * value) / (1.0 + std::exp(-c * value));
    }

}

double sigmoidDiff(double c, double value) {
    return c * (sigmoid(c, value) * (sigmoid(c, value) - 1.0));
}


void Trainer::gradientUpdate(Training::Position &pos) {
    //one step of stochastic gradient descent
    Board board;
    board.getPosition().BP = pos.bp();
    board.getPosition().WP = pos.wp();
    board.getPosition().K = pos.k();
    board.getPosition().color = (pos.mover() == Training::BLACK) ? BLACK : WHITE;

    if (pos.result() == Training::UNKNOWN)
        return;

    Line local;
    Value qStatic = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0);
    if (isWin(qStatic))
        return;
    auto mover = static_cast<double>(board.getPosition().getColor());
    double result = getWinValue(pos.result());
    double c = getCValue();

    double qValue = mover * static_cast<double>(qStatic);
    auto offset = static_cast<double>(scalfac * 1);
    for (size_t p = 0; p < 2; ++p) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8 * j + i);
                size_t index = 18ull * getIndex(curRegion, board.getPosition()) + 9ull * p + 3ull * j + i;
                double qDiffValue = mover * Trainer::evaluatePosition(board, gameWeights, index, offset);
                double qDiffValue2 = mover * Trainer::evaluatePosition(board, gameWeights, index, -offset);
                double diff = ((sigmoid(c, qValue) - result) *
                               sigmoidDiff(c, qValue) * (qDiffValue - qDiffValue2)) / (2.0 * offset);
                gameWeights[index] = gameWeights[index] - getLearningRate() * diff;

            }
        }
    }
    for (size_t index = SIZE; index < SIZE + 2; ++index) {
        double qDiffValue = mover * Trainer::evaluatePosition(board, gameWeights, index, offset);
        double qDiffValue2 = mover * Trainer::evaluatePosition(board, gameWeights, index, -offset);
        double diff = ((sigmoid(c, qValue) - result) *
                       sigmoidDiff(c, qValue) * (qDiffValue - qDiffValue2)) / (2.0 * offset);
        gameWeights[index] = gameWeights[index] - getLearningRate() * diff;
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
                      if ((counter % 200000) == 0) {
                          gameWeights.storeWeights("current.weights");
                          std::cout << "NonZero: " << gameWeights.numNonZeroValues() << std::endl;
                          std::cout << "Max: " << gameWeights.getMaxValue() << std::endl;
                          std::cout << "Min: " << gameWeights.getMinValue() << std::endl;
                          std::cout << "kingScore:" << gameWeights.kingOp << " | " << gameWeights.kingEnd << std::endl;
                          std::cout << std::endl;
                          std::cout << std::endl;
                      }
                  });
}


void Trainer::startTune() {
    int counter = 1;
    std::cout << "Data_size: " << data.positions_size() << std::endl;
    while (counter < getEpochs()) {
        std::cout << "CValue: " << getCValue() << std::endl;
        double loss = calculateLoss();
        if (loss > last_loss_value) {
            learningRate = 0.5 * learningRate;
            std::cout << "Dropped learning rate" << std::endl;
        }

        last_loss_value = loss;
        std::cout << "Loss: " << loss << std::endl;
        epoch();
        counter++;
        std::string name = "X" + std::to_string(counter) + ".weights";
        gameWeights.storeWeights(name);
        std::cout << "Stored weights" << std::endl;
        std::cout << "NonZero: " << gameWeights.numNonZeroValues() << std::endl;
        std::cout << "Max: " << gameWeights.getMaxValue() << std::endl;
        std::cout << "Min: " << gameWeights.getMinValue() << std::endl;
        std::cout << "kingScore:" << gameWeights.kingOp << " | " << gameWeights.kingEnd << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
    }
}

double Trainer::calculateLoss() {
    auto evalLambda = [this](const Training::Position &pos) {
        if (pos.result() == Training::UNKNOWN)
            return 0.0;
        Board board;
        board.getPosition().BP = pos.bp();
        board.getPosition().WP = pos.wp();
        board.getPosition().K = pos.k();
        board.getPosition().color = ((pos.mover() == Training::BLACK) ? BLACK : WHITE);


        double current = getWinValue(pos.result());
        auto color = static_cast<double>(board.getPosition().getColor());
        Line local;
        double quiesc = color * static_cast<double>(quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0));
        current = current - sigmoid(cValue, quiesc);
        current = current * current;
        return current;
    };
    double result = 0.0;
    std::for_each(data.mutable_positions()->begin(), data.mutable_positions()->end(), [&](Training::Position &pos) {
        result += evalLambda(pos);
    });

    return std::sqrt(result / static_cast<double>(data.positions_size()));
}

double Trainer::evaluatePosition(Board &board, Weights<double> &weights, size_t index, double offset) {
    weights[index] += offset;
    Line local;
    Value qDiff = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0);
    weights[index] -= offset;
    auto qDiffValue = static_cast<double>(qDiff);

    return qDiffValue;
}

