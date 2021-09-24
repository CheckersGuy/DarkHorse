//
// Created by root on 06.01.21.
//

#include "PyHelper.h"

void create_input(uint32_t w, uint32_t b, uint32_t k, int color, float *input) {
    Position p;
    p.WP = w;
    p.BP = b;
    p.K = k;
    p.color = ((color == -1) ? BLACK : WHITE);
    if (p.color == BLACK) {
        p = p.getColorFlip();
    }
    uint32_t white_men = p.WP & (~p.K);
    uint32_t black_men = p.BP & (~p.K);
    uint32_t white_kings = p.K & p.WP;
    uint32_t black_kings = p.K & p.BP;


    size_t offset = 0u;
    while (white_men != 0u) {
        auto index = Bits::bitscan_foward(white_men);
        white_men &= white_men - 1u;
        input[offset + index - 4] = 1;
    }
    offset += 28;
    while (black_men != 0u) {
        auto index = Bits::bitscan_foward(black_men);
        black_men &= black_men - 1u;
        input[offset + index] = 1;
    }
    offset += 28;
    while (white_kings != 0u) {
        auto index = Bits::bitscan_foward(white_kings);
        white_kings &= white_kings - 1u;
        input[offset + index] = 1;
    }
    offset += 32;
    while (black_kings != 0u) {
        auto index = Bits::bitscan_foward(black_kings);
        black_kings &= black_kings - 1u;
        input[offset + index] = 1;
    }
}

void deallocate_input(float *pointer) {
    free(pointer);
}

extern "C" int num_pieces(uint32_t white_men, uint32_t black_men, uint32_t kings) {
    return Bits::pop_count(white_men) + Bits::pop_count(black_men);
}
extern "C" int get_bucket_index(uint32_t white_men, uint32_t black_men, uint32_t kings) {
    auto num_pieces = Bits::pop_count(white_men) + Bits::pop_count(black_men);
    int index;
    if (num_pieces > 20) {
        index = 0;
    } else if (num_pieces > 16) {
        index = 1;
    } else if (num_pieces > 12) {
        index = 2;
    } else if (num_pieces > 8) {
        index = 3;
    } else if (num_pieces > 4) {
        index = 4;
    } else {
        index = 5;
    }
    int sub_index = ((kings == 0) ? 0 : 1);
    return 2 * index + sub_index;
}

extern "C" int invert_pieces(uint32_t pieces) {
    return static_cast<int>(getMirrored(pieces));
}

extern "C" int has_jumps(uint32_t white_men, uint32_t black_men, uint32_t kings, int color) {
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
    pos.color = (color == 1) ? WHITE : BLACK;
    std::cout << pos.get_fen_string() << std::endl;
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



