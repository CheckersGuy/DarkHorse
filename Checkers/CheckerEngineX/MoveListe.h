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

private:
    int moveCounter = 0;
    std::array<Move, 40> liste;
public:

    MoveListe() = default;

    int length() const;

    void addMove(Move next);

    void sort(Move ttMove, bool inPVLine, Color color);

    bool isEmpty() const;

    auto begin();

    auto end();

    auto begin() const;

    auto end() const;

    const Move &operator[](int index) const;

    Move &operator[](int index);

    void putFront(const Move& other);

};

inline bool MoveListe::isEmpty() const {
    return this->moveCounter == 0;
}

inline const Move &MoveListe::operator[](int index) const {
    return liste[index];
}

inline Move &MoveListe::operator[](int index) {
    return liste[index];
}

inline void MoveListe::addMove(Move next) {
    assert(!next.isEmpty());
    liste[this->moveCounter++] = next;
}

inline int MoveListe::length() const {
    return moveCounter;
}

inline auto MoveListe::begin() {
    return liste.begin();
}

inline auto MoveListe::end() {
    return liste.begin() + moveCounter;
}

inline auto MoveListe::begin() const {
    return liste.cbegin();
}

inline auto MoveListe::end() const {
    return liste.cbegin() + moveCounter;
}

#endif //CHECKERSTEST_MOVELISTE_H
