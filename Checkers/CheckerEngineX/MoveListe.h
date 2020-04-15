#ifndef CHECKERSTEST_MOVELISTE_H
#define CHECKERSTEST_MOVELISTE_H

#include <cstdio>
#include "Move.h"
#include <cassert>
#include "types.h"
#include "MovePicker.h"
#include <array>
#include <iterator>
#include "Position.h"
class MoveListe {

private:
    uint32_t moveCounter = 0;
    std::array<int16_t ,40> scores;

    void help_sort(int start_index);

public:
    std::array<Move, 40> liste;

    uint32_t length() const;

    void addMove(Move next);

    void sort(Move ttMove, bool inPVLine, Color color);

    void sort_static(Color mover, const Position &pos,const Move& ttMove);

    bool isEmpty() const;

    const Move &operator[](int index) const;

    Move &operator[](int index);

    void putFront(const Move& other);

    auto begin(){
        return liste.begin();
    }

    auto end(){
        return liste.end();
    }

    MoveListe& operator=(const MoveListe& other);

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
    liste[this->moveCounter++] = next;
}

inline uint32_t MoveListe::length() const {
    return moveCounter;
}



#endif //CHECKERSTEST_MOVELISTE_H