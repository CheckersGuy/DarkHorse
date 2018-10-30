//
// Created by robin on 8/1/18.
//

#include "Trainer.h"


Value WorkerPool::operator[](int index) {
    return results[index];
}


void WorkerPool::waitAll() {
    while(!work.empty()){
        std::cout<<work.size()<<"\n";
    }
}

void WorkerPool::startThreads() {
    for(int i=0;i<threads;++i){
        std::cout<<"Started thread "<<i<<"\n";
        workers.push_back(std::thread(idleLoop,this));
    }
}

void WorkerPool::addWork(Position pos) {
    this->work.push(pos);
}

void WorkerPool::setOutput(Value *val) {
    this->results =val;
}

void WorkerPool::idleLoop(WorkerPool *pool) {

    while(!pool->work.empty()){
        pool->myMutex.lock();
        Position work=pool->work.front();
        pool->work.pop();

        pool->myMutex.unlock();

        Board current;
        BoardFactory::setUpPosition(current,work);
        Value val = Training::quiescene(*pool->weights,current,-INFINITE,INFINITE,0,100000000);
        pool->results[pool->gameCounter]=val;
        pool->gameCounter++;
    }
    std::cout<<"Stopped the loop: "<<pool->work.size()<<"\n";
}

int Trainer::getThreads() {
    return threads;
}

int Trainer::getEpochs() {
    return epochs;
}

void Trainer::setThreads(int threads) {
    this->threads = threads;
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
    l2Reg=reg;
}

void Trainer::setLearningRate(double learn) {
    learningRate=learn;
}

float getWinValue(Score score) {
    if (score == BLACK_WIN)
        return 0.0;
    else if (score == WHITE_WIN)
        return 1.0;


        return 0.5;
}


void Trainer::epoch() {
    //Doing one epoch
    for(Training::TrainingPos position : data.positions){
        double result =getWinValue(position.result);
        double color =static_cast<double>(position.pos.color);
        Board board;
        BoardFactory::setUpPosition(board,position.pos);
        double quiesce =static_cast<double>(Training::quiescene(weights,board,-INFINITE,INFINITE,0,100000000).value);
        double diff =(Training::sigmoid(getCValue(),color*quiesce)-result)*Training::sigmoidDiff(getCValue(),color*quiesce);
        for(int p=0;p<2;++p){
            for(int j=0;j<3;++j){
                for(int i=0;i<3;++i){
                    const uint32_t curRegion = region << (8*j+i);
                    int index = getIndex(curRegion, position.pos)+390625*(3*j+i)+390625*9*p;
                    weights.weights[index]+=1.0;
                    double quieDiff =static_cast<double>(Training::quiescene(weights,board,-INFINITE,INFINITE,0,100000000).value);
                    weights.weights[index]-=1.0;
                    quieDiff =color*(quieDiff-quiesce);
                    double currDiff =diff*quieDiff+l2Reg*weights.weights[index];
                    weights.weights[index]=weights.weights[index]-learningRate*currDiff;
                }
            }
        }

    }
}

void Trainer::startTune() {

    int counter=0;

    while(counter<getEpochs()){
        data.shuffle();
        std::cout<<"Epoch ";
        std::cout<<"MaxValue: "<<weights.getMaxValue()<<"\n";
        std::cout<<"MinValue: "<<weights.getMinValue()<<"\n";
        std::cout<<"NonZero: "<<weights.numNonZeroValues()<<"\n";
        epoch();
        if((counter%5)==0){
            double loss = calculateLoss();
            std::cout << "Loss: " << loss << std::endl;
        }
        std::string name ="a"+std::to_string(counter)+".weights";
        weights.storeWeights(name);
        counter++;
        std::cout<<"\n";
        std::cout<<"\n";

    }
}

double Trainer::calculateLoss() {
    double mse = 0;

    for (Training::TrainingPos pos : data.positions) {
        Board board;
        BoardFactory::setUpPosition(board, pos.pos);
        double current = getWinValue(pos.result);
        double color =static_cast<double>(pos.pos.color);
        double quiesc =color*static_cast<double>(Training::quiescene( this->weights,board, -INFINITE, INFINITE, 0,100000000).value);
        current = current - Training::sigmoid(cValue, quiesc);
        current = current * current;
        mse += current;
    }
    mse = mse / static_cast<double>(data.length());
    mse = std::sqrt(mse);
    return mse;
}


