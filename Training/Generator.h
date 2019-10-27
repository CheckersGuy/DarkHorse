//
// Created by robin on 7/26/18.
//

#ifndef TRAINING_GENERATOR_H
#define TRAINING_GENERATOR_H

#include <string>
#include <fstream>
#include <unordered_set>
#include "Training.h"
#include "Engine.h"
#include  <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include  <sys/types.h>
#include "fcntl.h"
#include "Utilities.h"


using namespace Training;

class Generator {

private:

    std::unordered_set<TrainingGame,Training::TrainingHash,Training::TrainingComp>buffer;
    const std::string book;
    const std::string output;
    Engine &engine;
    int threads, maxGames, time;
public:

    Generator(Engine &engine, std::string book, std::string output) : engine(engine), book(book), output(output),
                                                                      time(100), maxGames(100) {};


    void setMaxGames(int games);

    void setThreads(int threads);

    void clearBuffer();

    void start();

    void setTime(int time);

    int getTime();

    int getThreads();

    int getMaxGames();


};


#endif //TRAINING_GENERATOR_H
