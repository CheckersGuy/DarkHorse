//
// Created by root on 06.01.21.
//

#include "PyHelper.h"






inline size_t getIndex2(uint32_t reg, const Position &pos) {
    //trying to be a little bit more efficient with the pext instruction
    uint32_t orig_pieces = (pos.BP | pos.WP) & reg;
    uint32_t pieces = (pos.BP | pos.WP);
    pieces = _pext_u32(pieces, reg);

    uint32_t BP = pos.BP & (~pos.K);
    uint32_t WP = pos.WP & (~pos.K);
    uint32_t BK = pos.BP & pos.K;
    uint32_t WK = pos.WP & pos.K;
    size_t index = 0ull;
    while (pieces) {
        uint32_t lsb = (orig_pieces & ~(orig_pieces - 1u));
        size_t temp_index = Bits::bitscan_foward(pieces);
        size_t current = ((BP & lsb) != 0u) * 3ull + ((WP & lsb) != 0u) * 4ull + ((BK & lsb) != 0u) * 1ull +
                         ((WK & lsb) != 0u) * 2ull;
        index += current * powers[temp_index];
        pieces &= pieces - 1u;
        orig_pieces &= orig_pieces - 1u;
    }

    return index;
}

extern "C" void print_position(uint32_t white_men, uint32_t black_men, uint32_t kings) {
    Position pos;
    pos.WP = white_men;
    pos.BP = black_men;
    pos.K = kings;
    pos.printPosition();
}

extern "C" int count_bits(uint32_t bitfield) {
    return __builtin_popcount(bitfield);
}

extern "C" int count_white_kings(uint32_t white_men, uint32_t black_men, uint32_t kings) {
    return __builtin_popcount(kings & white_men);
}

extern "C" int count_black_kings(uint32_t white_men, uint32_t black_men, uint32_t kings) {
    return __builtin_popcount(kings & black_men);
}

extern "C" int count_white_pawn(uint32_t white_men, uint32_t black_men, uint32_t kings) {
    return __builtin_popcount((~kings) & white_men);
}

extern "C" int count_black_pawn(uint32_t white_men, uint32_t black_men, uint32_t kings) {
    return __builtin_popcount((~kings) & black_men);
}

extern "C" int get_pattern_index(int i, int j, int p, uint32_t white_men, uint32_t black_men, uint32_t kings) {
    //we can get the endgame index simply by adding 'pattern_size' which will be 9 in our case

    constexpr uint32_t region = 13107u;

    Position pos;
    pos.WP = white_men;
    pos.BP = black_men;
    pos.K = kings;

    const uint32_t curRegion = region << (8u * j + i);
    const auto region_index = getIndex2(curRegion, pos);
    return 18 * (int) region_index + 3 * j + i + 9 * p;
}

