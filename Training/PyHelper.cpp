//
// Created by root on 06.01.21.
//

#include "PyHelper.h"


extern "C" int get_bucket_index(uint32_t white_men, uint32_t black_men, uint32_t kings){
    auto num_pieces = Bits::pop_count(white_men)+Bits::pop_count(black_men);

    if(num_pieces>=8){
        return 0;
    }
    return 1;
}

extern "C" int invert_pieces(uint32_t pieces){
    return static_cast<int>(getMirrored(pieces));
}

extern "C" int has_jumps(uint32_t white_men, uint32_t black_men, uint32_t kings,int color){
    Position pos;
    pos.BP = black_men;
    pos.WP = white_men;
    pos.K = kings;
    return pos.hasJumps(static_cast<Color>(color));
}


extern "C" void print_fen(int color, uint32_t white_men, uint32_t black_men, uint32_t kings) {
    Position pos;
    pos.BP = black_men;
    pos.WP = white_men;
    pos.K = kings;
    pos.color =(color == 1)?WHITE : BLACK;
    std::cout<<pos.get_fen_string()<<std::endl;
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



