//
// Created by robin on 7/18/18.
//

#include "Engine.h"

Engine::Engine(const std::string myPath) : path(myPath), hashSize(DEFAULT_HASH) {

    handle = dlopen(myPath.c_str(), RTLD_LAZY);
    if (!handle) {
        std::cerr << "Engine wasn't found" << std::endl;
        exit(EXIT_FAILURE);
    }
};


Engine::~Engine() {
    dlclose(handle);
}

Value Engine::searchEngine(Board &board, Move &best, int depth, int time, bool flag) {
    //searches
    auto myfunc = (Search) (dlsym(handle, "searchValue"));
    Value value = myfunc(board, best, depth, time, flag);
    return value;
}

void Engine::initialize() {
    Init func = (Init) (dlsym(handle, "initialize"));
    if (!func) {
        std::cout << "Couldn't find the function";
        exit(EXIT_FAILURE);
    }
    func();
    setHashSize(DEFAULT_HASH);
}

void Engine::setHashSize(int hash) {
    hashSize = hash;
    HashSize func = (HashSize) (dlsym(handle, "setHashSize"));
    if (!func) {
        std::cout << "Couldn't find the function";
        exit(EXIT_FAILURE);
    }
    func(hash);

}

void Engine::setTimePerMove(int time) {
    this->timePerMove=time;
}

int Engine::getTimePerMove() {
    return this->timePerMove;
}

