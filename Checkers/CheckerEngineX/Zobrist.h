//
// Created by Robin on 06.05.2018.
//

#ifndef CHECKERENGINEX_ZOBRIST_H
#define CHECKERENGINEX_ZOBRIST_H

#include <cstdint>
#include <random>
#include <iostream>
#include "Position.h"
#include "types.h"
#include "types.h"


namespace Zobrist {
    extern std::array<std::array<uint64_t ,4>,32> ZOBRIST_KEYS;
    extern uint64_t colorBlack;

    void initializeZobrisKeys();

    uint64_t generateKey(const Position &pos);

    void doUpdateZobristKey(Position &pos, Move move);
}


#endif //CHECKERENGINEX_ZOBRIST_H
