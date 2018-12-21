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
    if (score == BLACK_WIN)
        return 0.0;
    else if (score == WHITE_WIN)
        return 1.0;


    return 0.5;
}


void Trainer::epoch() {
    for (Training::TrainingPos position : data.positions) {
        Board board;
        BoardFactory::setUpPosition(board, position.pos);
        Line local;
        Value qStatic = quiescene<NONPV>(board, -INFINITE, INFINITE,local, 0);
       if (qStatic.isWinning()) {
            continue;

        }

        double result = getWinValue(position.result);
        size_t colorIndex = 0;
        if (board.getMover() == WHITE) {
            colorIndex = 390625 * 9 * 2;
        }
       double c =getCValue();
       double mover =static_cast<double>(board.getMover());
        for (size_t p = 0; p < 2; ++p) {
            for (size_t j = 0; j < 3; ++j) {
                for (size_t  i = 0; i < 3; ++i) {
                    const uint32_t curRegion = region << (8 * j + i);
                    size_t index = getIndex(curRegion, position.pos) + 390625 * (3 * j + i) + 390625 * 9 * p+colorIndex;
                    double qValue= qStatic.as<double>();
                    gameWeights.weights[index]+=50.0;
                    Line local;
                    Value qDiff = quiescene<NONPV>( board, -INFINITE, INFINITE,local, 0);
                    gameWeights.weights[index]-=50.0;
                    double qDiffValue= qDiff.as<double>();
                    double diff =mover*((Training::sigmoid(c,mover*qValue)-result)*Training::sigmoidDiff(c,mover*qValue)*(qDiffValue-qValue))/50.0;

                    diff+=gameWeights.weights[index]*getL2Reg();
                    gameWeights.weights[index]=gameWeights.weights[index]-getLearningRate()*diff;

                }
            }
        }
    }
}

void Trainer::startTune() {

    int counter = 0;

    while (counter < getEpochs()) {
        data.shuffle();
        std::cout << "Epoch ";
        std::cout << "MaxValue: " << gameWeights.getMaxValue() << "\n";
        std::cout << "MinValue: " << gameWeights.getMinValue() << "\n";
        std::cout << "NonZero: " << gameWeights.numNonZeroValues() << "\n";

        epoch();
        if ((counter % 4) == 0) {
            double loss = calculateLoss();
            std::cout << "Loss: " << loss << std::endl;
        }
        std::string name = "a" + std::to_string(counter) + ".weights";
        gameWeights.storeWeights(name);
        counter++;
        std::cout << "\n";
        std::cout << "\n";

    }
}

double Trainer::calculateLoss() {
    double mse = 0;

    for (Training::TrainingPos pos : data.positions) {
        Board board;
        BoardFactory::setUpPosition(board, pos.pos);
        double current = getWinValue(pos.result);
        double color = static_cast<double>(pos.pos.color);
        Line local;
        double quiesc = color * static_cast<double>(quiescene<NONPV>( board, -INFINITE, INFINITE,local, 0).value);
        current = current - Training::sigmoid(cValue, quiesc);
        current = current * current;
        mse += current;
    }
    mse = mse / static_cast<double>(data.length());
    mse = std::sqrt(mse);
    return mse;
}




