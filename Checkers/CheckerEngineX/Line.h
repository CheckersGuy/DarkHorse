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
    int counter=0;

    Line()=default;

    Line(const Line &other);

    int length();

    void addMove(Move move);

    void concat(Move best, Line &line);

    std::string toString();

    Move getFirstMove();

    void clear();

    bool operator==(Line &other);

    bool operator!=(Line &other);

    Move operator[](int index);

};

inline Move Line::operator[](int index) {
    return this->myArray[index];
}

std::ostream& operator<<(std::ostream& stream, Line& line);

#endif //CHECKERENGINEX_LINE_H
