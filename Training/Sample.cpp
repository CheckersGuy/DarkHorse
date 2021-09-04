//
// Created by robin on 01.09.21.
//

#include "Sample.h"


bool Sample::operator==(const Sample &other) const {
    return (position == other.position && result == other.result);
}

bool Sample::operator!=(const Sample &other) const {
    return !((*this) == other);
}

std::ostream &operator<<(std::ostream &stream, const Sample s) {
    stream.write((char *) &s.position, sizeof(Position));
    stream.write((char *) &s.result, sizeof(int));
    return stream;
}

std::istream &operator>>(std::istream &stream, Sample &s) {
    Position pos;
    stream.read((char *) &pos, sizeof(Position));
    int result;
    stream.read((char *) &result, sizeof(int));
    s.result = result;
    s.position = pos;
    return stream;
}