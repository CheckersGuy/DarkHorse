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
    int win, loss, draw;
    int threads;
    std::string openingBook;

public:

    Match() = default;

    Match(Engine &first, Engine &second) : first(first), second(second), win(0), loss(0), draw(0), maxGames(1000),
                                           time(100), threads(1), openingBook("Positions/3move.pos") {};

    void setMaxGames(int games);

    void setTime(int time);

    int getMaxGames();

    int getTime();

    void initializeEngines();

    void start();

    int getWin();

    int getLoss();

    int getDraw();

    int getElo();

    void setNumThreads(int threads);

    int getNumThreads();

    std::string &getOpeningBook();

    void setOpeningBook(std::string book);


};


#endif //TRAINING_MATCH_H
