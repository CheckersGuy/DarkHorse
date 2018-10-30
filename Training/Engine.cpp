//
// Created by robin on 7/18/18.
//

#include "Engine.h"

Engine::~Engine() {
    dlclose(handle);
}

Value Engine::searchEngine(Board &board, Move &best, int depth, int time, bool flag) {
    //searches
    Search myfunc = (Search) (dlsym(handle, "searchValue"));
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

