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


double getWinValue(Score score) {
    if (score < 0)
        return 0.0;
    else if (score > 0)
        return 1.0;

    return 0.5;
}


void Trainer::gradientUpdate(TrainingPos position) {
    //one step of stochastic gradient descent
    Board board;
    board = position.pos;
    Line local;
    Value qStatic = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0);
    auto mover = static_cast<double>(position.pos.getColor());
    double result = getWinValue(position.result);
    double c = getCValue();

    double offset = static_cast<double>(scalfac);
    double qValue = mover * static_cast<double>(qStatic);
    for (size_t p = 0; p < 2; ++p) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8 * j + i);
                size_t index = 18ull * getIndex(curRegion, position.pos) + 9ull * p + 3ull * j + i;
                double qDiffValue = mover * Trainer::evaluatePosition(board, gameWeights, index, offset);
                double qDiffValue2 = mover * Trainer::evaluatePosition(board, gameWeights, index, -offset);
                double diff = ((Training::sigmoid(c, qValue) - result) *
                               Training::sigmoidDiff(c, qValue) * (qDiffValue - qDiffValue2)) / (2.0 * offset);
                gameWeights[index] = gameWeights[index] - getLearningRate() * diff;

            }
        }
    }
    for (size_t index = SIZE; index < SIZE + 2; ++index) {
        double qDiffValue = mover * Trainer::evaluatePosition(board, gameWeights, index, offset);
        double qDiffValue2 = mover * Trainer::evaluatePosition(board, gameWeights, index, -offset);
        double diff = ((Training::sigmoid(c, qValue) - result) *
                       Training::sigmoidDiff(c, qValue) * (qDiffValue - qDiffValue2)) / (2.0 * offset);
        gameWeights[index] = gameWeights[index] - getLearningRate() * diff;
    }


}


void Trainer::epoch() {
    std::cout << "Start shuffling" << std::endl;
    std::shuffle(data.begin(), data.end(), generator);
    std::cout << "Done shuffling" << std::endl;
    int counter = 0;
    std::for_each(data.begin(), data.end(), [this, &counter](TrainingPos pos) {
        counter++;
        gradientUpdate(pos);

        if ((counter % 200000) == 0) {
            gameWeights.storeWeights("test.weights");
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
    while (counter < getEpochs()) {
        std::cout << "CValue: " << getCValue() << std::endl;
        double loss = 0;
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
    auto evalLambda = [this](TrainingPos pos) {
        Board board;
        board = pos.pos;
        double current = getWinValue(pos.result);
        auto color = static_cast<double>(pos.pos.color);
        Line local;
        double quiesc = color * static_cast<double>(quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0));
        current = current - Training::sigmoid(cValue, quiesc);
        current = current * current;
        return current;
    };
    //0.288253
    double result = 0.0;
    for (auto &pos : data) {
        result += evalLambda(pos);
    }

    return std::sqrt(result / static_cast<double>(data.size()));
}

double Trainer::evaluatePosition(Board &board, Weights<double> &weights, size_t index, double offset) {
    weights[index] += offset;
    Line local;
    Value qDiff = quiescene<PVNode>(board, -INFINITE, INFINITE, local, 0);
    weights[index] -= offset;
    auto qDiffValue = static_cast<double>(qDiff);

    return qDiffValue;
}

