//
// Created by robin on 9/17/18.
//

#ifndef CHECKERENGINEX_LINE_H
#define CHECKERENGINEX_LINE_H


#include "Move.h"
#include "types.h"

class Line {

public:

    Move myArray[MAX_PLY];
    int counter = 0;

    Line() = default;

    Line(const Line &other);

    int length() const;

    void addMove(Move move);

    void concat(Move best, Line &line);

    std::string toString() const;

    Move getFirstMove() const;

    void clear();

    bool operator==(const Line &other) const;

    bool operator!=(const Line &other) const;

    Line& operator=(const Line& other);

    Move operator[](int index) const;

    Move &operator[](int index);
};


std::ostream &operator<<(std::ostream &stream, Line &line);

#endif //CHECKERENGINEX_LINE_H
