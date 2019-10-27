//
// Created by robin on 9/17/18.
//

#ifndef CHECKERENGINEX_LINE_H
#define CHECKERENGINEX_LINE_H


#include "Move.h"
#include "types.h"
class Line {

private:
    std::array<Move, MAX_PLY> myArray;
    int counter = 0;

public:

    Line() = default;

    Line(const Line &other) noexcept;

    int length() const;

    void addMove(const Move &move);

    void concat(const Move &best, const Line &line);

    std::string toString() const;

    Move &getFirstMove();

    void clear();

    bool operator==(const Line &other) const;

    bool operator!=(const Line &other) const;

    Line &operator=(const Line &other);

    const Move &operator[](int index) const;

    Move &operator[](int index);

    auto begin();

    auto end();

};

inline auto Line::begin() {
    return myArray.begin();
}

inline auto Line::end() {
    return myArray.begin() + counter;
}

std::ostream &operator<<(std::ostream &stream, Line &line);

#endif //CHECKERENGINEX_LINE_H
