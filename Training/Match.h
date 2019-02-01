//
// Created by robin on 7/26/18.
//

#ifndef TRAINING_MATCH_H
#define TRAINING_MATCH_H

#include "Engine.h"
#include  <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include  <sys/types.h>
#include "fcntl.h"
#include "Utilities.h"
class Match {

private:
    Engine &first;
    Engine &second;
    int time;
    int maxGames;
    int wins, losses, draws;
    int threads;
    std::string openingBook;

public:

    Match() = default;

    Match(Engine &first, Engine &second) : first(first), second(second), wins(0), losses(0), draws(0), maxGames(1000),
                                           time(100), threads(1), openingBook("Positions/3move.pos") {};

    void setMaxGames(int games);

    int getMaxGames();

    void initializeEngines();

    void start();

    int getWins();

    int getLosses();

    int getDraws();

    int getElo();

    void setNumThreads(int threads);

    int getNumThreads();

    std::string &getOpeningBook();

    void setOpeningBook(std::string book);


};


#endif //TRAINING_MATCH_H
