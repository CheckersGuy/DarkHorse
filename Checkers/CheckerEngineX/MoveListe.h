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


class MoveListIterator {

friend class MoveListe;
public:
    using value_type =Move;
    using difference_type=std::ptrdiff_t;
    using pointer=Move *;
    using reference=Move &;
    using iterator_category =std::random_access_iterator_tag;

    MoveListIterator() = default;

    MoveListIterator(const MoveListIterator &other) {
        this->p = other.p;
    }

    bool operator==(const MoveListIterator &iter) const {
        return iter.p == p;
    }

    bool operator!=(const MoveListIterator &iter) const {
        return iter.p != p;
    }

    value_type& operator*() const {
        return *p;
    }

    pointer operator->(){
        return p;
    }

    MoveListIterator &operator++() {
        p++;
        return *this;
    }

    MoveListIterator operator++(int) {
        MoveListIterator iter;
        iter.p = p;
        p++;
        return iter;
    }

    MoveListIterator &operator--() {
        p--;
        return *this;
    }

    MoveListIterator operator--(int) {
        MoveListIterator iter;
        iter.p = p;
        p--;
        return iter;
    }

    MoveListIterator &operator+=(const difference_type dist) {
        this->p += dist;
        return *this;
    }

    MoveListIterator &operator-=(const difference_type dist) {
        this->p -= dist;
        return *this;
    }

    MoveListIterator operator+(const difference_type dist) const {
        MoveListIterator iter;
        iter.p = this->p + dist;
        return iter;
    }

    difference_type operator-(const MoveListIterator &iter) const {
        difference_type dist = this->p - iter.p;
        return dist;
    }

    MoveListIterator operator-(const difference_type dist) const {
        MoveListIterator next;
        next.p = this->p;
        next.p -= dist;
        return next;
    }

    value_type operator[](const difference_type index) {
        return *(this->p + index);
    }

    bool operator<(const MoveListIterator &iter) const {
        return p < iter.p;
    }

    bool operator<=(const MoveListIterator &iter) const {
        return p <= iter.p;
    }

    bool operator>(const MoveListIterator &iter) const {
        return p > iter.p;
    }

    bool operator>=(const MoveListIterator &iter) const {
        return p >= iter.p;
    }
    friend MoveListIterator operator+(difference_type lhs, const MoveListIterator &rhs) {
        MoveListIterator next;
        next.p = rhs.p + lhs;
        return next;
    }

    friend MoveListIterator operator-(difference_type lhs, const MoveListIterator &rhs) {
        MoveListIterator next;
        next.p = rhs.p - lhs;
        return next;
    }


private:
    pointer p;
};




class MoveListe {

public:
    int moveCounter = 0;
    Move liste[40];
    int scores[40];
    MoveListe() = default;

    int length();

    void addMove(Move& next);

    void swap(int a, int b);

    void swap(Move &a, Move &b);

    void sort(Move ttMove, bool inPVLine, Color color);

    Move operator[](size_t index);

    int findIndex(Move move);

    bool isInList(Move move);

    MoveListIterator begin();

    MoveListIterator end();

};


inline bool MoveListe::isInList(Move move) {
    return findIndex(move)>=0;
}

inline Move MoveListe::operator[](size_t index) {
    return liste[index];
}

inline void MoveListe::addMove(Move&next) {
    assert(!next.isEmpty());
    next.setMoveIndex(moveCounter);
    this->liste[this->moveCounter++] = next;
}

inline int MoveListe::length() {
    return moveCounter;
}

inline void MoveListe::swap(int a, int b) {
    const Move temp = liste[a];
    liste[a] = liste[b];
    liste[b] = temp;
}

inline void MoveListe::swap(Move &a, Move &b) {
    Move temp = a;
    a = b;
    b = temp;
}


#endif //CHECKERSTEST_MOVELISTE_H
