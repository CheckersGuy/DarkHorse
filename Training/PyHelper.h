//
// Created by root on 06.01.21.
//

#ifndef READING_PYHELPER_H
#define READING_PYHELPER_H

#include <cstdint>
#include "Position.h"

extern "C" void print_position(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int count_bits(uint32_t bitfield);
extern "C" int get_pattern_index(int i, int j, int p, uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int count_white_kings(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int count_black_kings(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int count_white_pawn(uint32_t white_men, uint32_t black_men, uint32_t kings);
extern "C" int count_black_pawn(uint32_t white_men, uint32_t black_men, uint32_t kings);
#endif //READING_PYHELPER_H
