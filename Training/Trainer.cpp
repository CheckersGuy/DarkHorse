//
// Created by robin on 8/1/18.
//

#include <future>
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



void Trainer::gradientUpdate(TrainingPos position) {
    //one step of stochastic gradient descent
    Board board;
    BoardFactory::setUpPosition(board, position.pos);
    Line local;
    Value qStatic = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0);
    double result = getWinValue(position.result);

    double c = getCValue();
    double mover = static_cast<double>(board.getMover());
    for (size_t p = 0; p < 2; ++p) {
        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8 * j + i);
                size_t index= 18 * getIndex(curRegion, position.pos) + 9 * p + 3 * j + i;

                double qValue = qStatic.as<double>();
                gameWeights.weights[index] += scalFac;
                Line local;
                Value qDiff = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0);
                gameWeights.weights[index] -= scalFac;
                double qDiffValue = qDiff.as<double>();
                double diff = mover * ((Training::sigmoid(c, mover * qValue) - result) *
                                       Training::sigmoidDiff(c, mover * qValue) * (qDiffValue - qValue)) ;

                diff += gameWeights.weights[index] * getL2Reg();
                gameWeights.weights[index] = gameWeights.weights[index] - getLearningRate() * diff;

            }
        }
    }
    for(size_t index=SIZE;index<SIZE+4;++index){
        double qValue = qStatic.as<double>();
        gameWeights[index] += scalFac;
        Line local;
        Value qDiff = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0);
        gameWeights[index] -= scalFac;
        double qDiffValue = qDiff.as<double>();
        double diff = mover * ((Training::sigmoid(c, mover * qValue) - result) *
                               Training::sigmoidDiff(c, mover * qValue) * (qDiffValue - qValue)) ;

        diff += gameWeights[index] * getL2Reg();
        gameWeights[index] = gameWeights[index] - getLearningRate() * diff;
    }


}


void Trainer::epoch() {
    std::cout << "Start shuffling" << std::endl;
    std::shuffle(data.begin(), data.end(), generator);
    std::cout << "Done shuffling" << std::endl;
    std::for_each(data.begin(), data.end(), [this](TrainingPos pos) { gradientUpdate(pos); });
}

void Trainer::epochC() {
    std::cout << "Start shuffling" << std::endl;
    std::shuffle(data.begin(), data.end(), generator);
    std::cout << "Done shuffling" << std::endl;
    std::for_each(data.begin(), data.end(), [this](TrainingPos position) {
        double result = getWinValue(position.result);
        Board board;
        BoardFactory::setUpPosition(board, position.pos);
        Line local;
        Value qStatic = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0);
        double mover= static_cast<double>(board.getMover());
        double qValue=static_cast<double>(qStatic.value)*mover;
        double diff=(Training::sigmoid(getCValue(),qValue)-result)*Training::sigmoidDiff(getCValue(),qValue)*qValue;
        cValue=cValue-learningRate*diff;
    });
}

void Trainer::startTune() {

    int counter = 1;

    while (counter < getEpochs()) {

        std::cout<<"CValue: "<<getCValue()<<std::endl;
        double loss = calculateLoss();
        std::cout << "Loss: " << loss << std::endl;
        epoch();

        counter++;

        std::string name = "D" + std::to_string(counter) + ".weights";
        gameWeights.storeWeights(name);
        std::cout <<"Stored weights" << std::endl;
        std::cout <<"NonZero: "<<gameWeights.numNonZeroValues()<<std::endl;
        std::cout<<"Max: "<<gameWeights.getMaxValue()<<std::endl;
        std::cout<<"Min: "<<gameWeights.getMinValue()<<std::endl;
        std::cout<<"balanceScore:"<<gameWeights.balanceOp<<" | "<<gameWeights.balanceEnd<<std::endl;
        std::cout<<"kingScore:"<<gameWeights.kingOp<<" | "<<gameWeights.kingEnd<<std::endl;
        std::cout << std::endl;
        std::cout << std::endl;

    }
}

void Trainer::startTuneC() {

}


double Trainer::calculateLoss(int threads) {
    double mse = 0;
    auto evalLambda = [this](TrainingPos pos) {
        Board board;
        BoardFactory::setUpPosition(board, pos.pos);
        double current = getWinValue(pos.result);
        double color = static_cast<double>(pos.pos.color);
        Line local;
        double quiesc = color * static_cast<double>(quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0).value);
        current = current - Training::sigmoid(cValue, quiesc);
        current = current * current;
        return current;
    };

    std::vector<std::future<double>> workers;

    size_t chunk = data.size() / threads;
    size_t i = 0;
    for (; i < data.size(); i += chunk) {
        workers.emplace_back(std::async(std::launch::async, [=]() {
            double local = 0;
            for (size_t k = i; k < i + chunk; ++k) {
                local += evalLambda(data[k]);
            }
            return local;
        }));
    }
    for (; i < data.size(); ++i) {
        mse += evalLambda(data[i]);
    }

    for (auto &th : workers) {
        mse += th.get();
    }
    return std::sqrt(mse / static_cast<double>(data.size()));
}
