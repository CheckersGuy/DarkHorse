#ifndef CHECKERSTEST_MOVELISTE_H
#define CHECKERSTEST_MOVELISTE_H

#include "Move.h"
#include "types.h"
#include "MovePicker.h"
#include <array>
#include "Position.h"

class MoveListe {

private:
    int moveCounter = 0;
    std::array<int16_t, 40> scores;
public:
    std::array<Move, 40> liste;

    int length() const;

    void addMove(Move next);

    void sort(Position current,int ply,Move ttMove, int start_index);

    bool isEmpty() const;

    const Move &operator[](int index) const;

    Move &operator[](int index);

    void putFront(Move other);

    void remove(Move move);


    void reset();

    auto begin() {
        return liste.begin();
    }

   int get_move_index(Move move) const;

    auto end() {
        auto it = liste.begin();
        std::advance(it, moveCounter);
        return it;
    }

    MoveListe &operator=(const MoveListe &other);

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

inline int MoveListe::length() const {
    return moveCounter;
}


#endif //CHECKERSTEST_MOVELISTE_H