//
// Created by robin on 01.09.21.
//

#include "Sample.h"

std::string result_to_string(Result result) {
    if(result == BLACK_WON)
        return "BLACK_WON";
    if(result == WHITE_WON)
        return "WHITE_WON";
    if(result == DRAW)
        return "DRAW";

    return "UNKNOWN";
}
bool Sample::operator==(const Sample &other) const {
    return (position == other.position && result == other.result);
}

bool Sample::operator!=(const Sample &other) const {
    return !((*this) == other);
}

std::ostream &operator<<(std::ostream &stream, const Sample s) {
    stream.write((char *) &s.position, sizeof(Position));
    stream.write((char *) &s.result, sizeof(Result));
    stream.write((char *) &s.move, sizeof(int));
    return stream;
}

std::istream &operator>>(std::istream &stream, Sample &s) {
    Position pos;
    stream.read((char *) &pos, sizeof(Position));
    Result result;
    stream.read((char *) &result, sizeof(Result));
    int move;
    stream.read((char *) &move, sizeof(int));
    s.result = result;
    s.position = pos;
    s.move = move;
    return stream;
}