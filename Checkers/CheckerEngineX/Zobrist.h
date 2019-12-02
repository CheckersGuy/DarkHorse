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
#include <array>
#include "Bits.h"


namespace Zobrist {
    extern uint64_t colorBlack;


    void initializeZobrisKeys();

    uint64_t generateKey(const Position &pos);

    void doUpdateZobristKey(Position &pos, Move move);
}


#endif //CHECKERENGINEX_ZOBRIST_H
