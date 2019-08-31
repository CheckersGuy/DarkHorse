//
// Created by Robin on 04.01.2018.
//

#ifndef CHECKERSTEST_MOVELISTE_H
#define CHECKERSTEST_MOVELISTE_H

#include <stdio.h>
#include "Move.h"
#include <cassert>
#include "types.h"
#include "MovePicker.h"




class MoveListe {

public:
    int moveCounter = 0;
    Move liste[40];
    int scores[40];
    MoveListe() = default;

    int length();

    void addMove(Move next);

    void sort(Move ttMove, bool inPVLine, Color color);

    bool isEmpty() const;

    Move& operator[](size_t index);

    Move operator[](size_t index)const;

    Move* begin();

    Move* end();

};


inline bool MoveListe::isEmpty() const {
    return this->moveCounter==0;
}

inline Move& MoveListe::operator[](size_t index) {
    return liste[index];
}

inline Move MoveListe::operator[](size_t index)const {
    return liste[index];
}

inline void MoveListe::addMove(Move next) {
    assert(!next.isEmpty());
    next.setMoveIndex(moveCounter);
    liste[this->moveCounter++] = next;
}

inline int MoveListe::length() {
    return moveCounter;
}


#endif //CHECKERSTEST_MOVELISTE_H
