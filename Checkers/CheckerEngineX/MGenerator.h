//
// Created by Robin on 11.05.2017.
//
#ifndef CHECKERSTEST_MGENERATOR_H
#define CHECKERSTEST_MGENERATOR_H

//#include "intrin.h"
#include "Board.h"
#include "immintrin.h"
#include "MoveListe.h"
#include "types.h"


inline void addPawnMove(uint32_t from, uint32_t to, uint32_t captures, MoveListe &liste) {
    Move nMove;
    nMove.setTo(to);
    nMove.setFrom(from);
    nMove.captures = captures;
    liste.addMove(nMove);
}

inline void addKingMove(uint32_t from, uint32_t to, uint32_t captures, MoveListe &liste) {
    Move nMove;
    nMove.setPieceType(1);
    nMove.setTo(to);
    nMove.setFrom(from);
    nMove.captures = captures;
    liste.addMove(nMove);
}


void addCapturesBlack(Position &board, MoveListe &liste);

void addCapturesWhite(Position &board, MoveListe &liste);

void addWhiteTestPawns(Position &pos, uint32_t orig, uint32_t current, uint32_t captures, MoveListe &liste);

void addWhiteTestKings(Position &pos, uint32_t orig, uint32_t current, uint32_t captures, MoveListe &liste);

void addBlackTestPawns(Position &pos, uint32_t orig, uint32_t current, uint32_t captures, MoveListe &liste);

void addBlackTestKings(Position &pos, uint32_t orig, uint32_t current, uint32_t captures, MoveListe &liste);

void getMoves(Board &board, MoveListe &liste);

void getMoves(Position& pos,MoveListe& liste);

void getCaptures(Board&board, MoveListe& liste);

#endif //CHECKERSTEST_MGENERATOR_H