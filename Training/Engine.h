//
// Created by robin on 7/18/18.
//

#ifndef TRAINING_ENGINE_H
#define TRAINING_ENGINE_H

#include "types.h"
#include "dlfcn.h"
#include <iostream>
#include "Board.h"
#include "Move.h"

using Search =Value (*)(Board &, Move &, int, uint32_t, bool);
using Init =void (*)();
using HashSize=void (*)(int);

class Engine {
private:
    constexpr static int DEFAULT_HASH = 21;
    const std::string path;
    int timePerMove;
    int hashSize;
    void *handle;

public:

    Engine(const std::string path);

    ~Engine();

    Value searchEngine(Board &board, Move &best, int depth, int time, bool flag);

    void initialize();

    void setHashSize(int hash);

    void setTimePerMove(int time);

    int getTimePerMove();

};


#endif //TRAINING_ENGINE_H
