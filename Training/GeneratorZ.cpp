//
// Created by root on 19.05.21.
//

#include "GeneratorZ.h"

std::ostream &operator<<(std::ostream &stream, const TrainSample &s) {

    stream << s.pos;
    stream.write((char *) &s.result, sizeof(int));
    stream.write((char *) &s.evaluation, sizeof(int));
    return stream;
}

std::istream &operator>>(std::istream &stream, TrainSample &s) {
    stream >> s.pos;
    int result;
    stream.read((char *) &result, sizeof(int));
    s.result = result;
    int eval;
    stream.read((char *) &eval, sizeof(int));
    s.evaluation = eval;
    return stream;
}
