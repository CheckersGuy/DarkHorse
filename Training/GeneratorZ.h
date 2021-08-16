//
// Created by root on 19.05.21.
//

#ifndef READING_GENERATORZ_H
#define READING_GENERATORZ_H


#include <cstdint>
#include <types.h>
#include <Position.h>
#include <condition_variable>
#include <Board.h>
#include <GameLogic.h>

struct PolySample1{
    int result;
    int evaluation;
    Position pos;
    int piece_moved;

    friend std::ostream &operator<<(std::ostream &stream, const PolySample1& s);

    friend std::istream &operator>>(std::istream& stream, PolySample1 &s);

};

struct TrainSample {
    int result;
    int evaluation;
    Position pos;

    friend std::ostream &operator<<(std::ostream &stream, const TrainSample& s);

    friend std::istream &operator>>(std::istream& stream, TrainSample &s);

};


#endif //READING_GENERATORZ_H
