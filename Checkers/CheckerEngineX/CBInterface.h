//
// Created by Robin on 02.02.2018.
//
//Here goes allo the code for communication with CheckerBoard !!!
#ifndef CHECKERENGINEX_CBINTERFACE_H
#define CHECKERENGINEX_CBINTERFACE_H
#include "GameLogic.h"
#include <stdio.h>
#include <cstring>

enum CBKONST {
    CBWHITE = 1,
    CBBLACK = 2,
    CBMAN = 4,
    CBKING = 8,
    CBFREE = 0,

};

struct CBmove {
    int notUsed;
};

inline bool isBitSet(int var, uint32_t bit) {
    const uint32_t maske = 1u << bit;
    return ((maske & static_cast<uint32_t >(var)) != 0);
}



extern "C" {

MAKRO int getmove(int board[8][8], int color, double maxtime, char str[1024], int *playnow, int info, int unused,
            struct CBmove *move);

MAKRO int enginecommand(char str[256], char reply[1024]);
}

#endif //CHECKERENGINEX_CBINTERFACE_H
