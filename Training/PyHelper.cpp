//
// Created by root on 06.01.21.
//

#include "PyHelper.h"
#include <PosStreamer.h>
#include <BatchProvider.h>

enum class IndexType {
    INNER, PROMO
};

std::unique_ptr<BatchProvider> streamer;

extern "C" int init_streamer(size_t buffer_size, size_t batch_size, char *file_path) {
    std::string path(file_path);
    std::cout << "Path: " << file_path << std::endl;
    if (streamer.get() == nullptr) {
        streamer = std::make_unique<BatchProvider>(path, buffer_size, batch_size);
    }
    return streamer->get_streamer().get_file_size();
}

extern "C" void get_next_batch(float *results, float *inputs) {
    if (streamer.get() == nullptr) {
        std::exit(-1);
    }
    streamer->next(results, inputs);
}


template<IndexType type>
size_t get_big_index(uint32_t reg, const Position &pos) {
    const uint32_t PROMO_SQUARES = PROMO_SQUARES_BLACK | PROMO_SQUARES_WHITE;
    size_t index = 0ull;
    uint32_t BP = pos.BP & (~pos.K);
    uint32_t WP = pos.WP & (~pos.K);
    uint32_t orig_pieces = (pos.BP | pos.WP) & reg & (~PROMO_SQUARES);
    uint32_t promo_region = reg & PROMO_SQUARES;

    uint32_t promo_pieces = (pos.BP | pos.WP) & PROMO_SQUARES & reg;
    uint32_t promo_index = Bits::pext(promo_pieces, promo_region);
    uint32_t pieces = Bits::pext(orig_pieces, reg & (~PROMO_SQUARES));

    while (pieces) {
        uint32_t lsb = (orig_pieces & ~(orig_pieces - 1u));
        size_t temp_index = Bits::bitscan_foward(pieces);
        size_t current = ((BP & lsb) != 0u) * 1ull + ((WP & lsb) != 0u) * 2ull;
        index += current * powers3[temp_index];
        pieces &= pieces - 1u;
        orig_pieces &= orig_pieces - 1u;
    }

    if constexpr(type == IndexType::INNER) {
        return index;
    } else {
        return 8 * index + promo_index;
    }
}


inline size_t getIndex2(uint32_t reg, const Position &pos) {
    uint32_t orig_pieces = (pos.BP | pos.WP) & reg;
    uint32_t pieces = (pos.BP | pos.WP);
    pieces = Bits::pext(pieces, reg);

    uint32_t BP = pos.BP & (~pos.K);
    uint32_t WP = pos.WP & (~pos.K);
    uint32_t BK = pos.BP & pos.K;
    uint32_t WK = pos.WP & pos.K;
    size_t index = 0ull;
    while (orig_pieces) {
        uint32_t lsb = (orig_pieces & ~(orig_pieces - 1u));
        size_t temp_index = Bits::bitscan_foward(pieces);
        size_t current = ((BP & lsb) != 0u) * 1ull + ((WP & lsb) != 0u) * 2ull + ((BK & lsb) != 0u) * 3ull +
                         ((WK & lsb) != 0u) * 4ull;

        index += current * powers5[temp_index];
        pieces &= pieces - 1u;
        orig_pieces &= orig_pieces - 1u;
    }

    return index;
}

void patterns_op(uint32_t w, uint32_t b, uint32_t king, int color, int64_t *input) {
    size_t counter;
    Position pos;
    pos.WP = w;
    pos.BP = b;
    pos.K = king;
    pos.color = ((color == -1) ? BLACK : WHITE);
    if (pos.getColor() == BLACK) {
        pos = pos.getColorFlip();
    }
    const size_t offset1 = 8ull * 157464ull;
    const size_t offset2 = 4ull * 531441ull + 8ull * 157464ull;
    if (pos.K == 0) {
        //FOR THE PROMO_SQUARES
        for (auto i: {0, 2}) {
            size_t temp = ((i == 0) ? 0 : 1);
            for (auto k = 0; k < 2; ++k) {
                Position test;
                const uint32_t sub_reg = big_region << (8 * i + k);
                test.BP = sub_reg;
                size_t index = get_big_index<IndexType::PROMO>(sub_reg, pos);
                size_t sub_index_op = 8 * index + 2 * k + 4 * temp;
                input[counter++] = sub_index_op;
            }
        }
        //FOR THE NON_PROMO_SQUARES
        for (auto k = 0; k < 2; ++k) {
            const uint32_t sub_reg = big_region << (8 * 1 + k);
            size_t index = get_big_index<IndexType::INNER>(sub_reg, pos);
            size_t sub_index_op = 4 * index + 2 * k;
            input[counter++] = sub_index_op + offset1;
        }
    } else {
        for (auto i = 0; i < 3; ++i) {
            for (auto k = 0; k < 3; ++k) {
                const uint32_t sub_reg = region << (8 * i + k);
                size_t index = getIndex2(sub_reg, pos);
                size_t sub_index_op = 18 * index + 2 * k + 6 * i;
                input[counter++] = sub_index_op + offset2;
            }
        }
    }
}


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

    //trying out some different buckets



    auto num_pieces = Bits::pop_count(white_men) + Bits::pop_count(black_men);
    int index;
    if (num_pieces > 20) {
        index = 0;
    } else if (num_pieces > 18) {
        index = 1;
    } else if (num_pieces > 16) {
        index = 2;
    } else if (num_pieces > 14) {
        index = 3;
    } else if (num_pieces > 12) {
        index = 5;
    } else if (num_pieces > 10) {
        index = 6;
    } else if (num_pieces > 8) {
        index = 7;
    } else if (num_pieces > 4) {
        index = 8;
    } else if (num_pieces > 2) {
        index = 9;
    } else {
        index = 10;
    }
    if (kings == 0) {
        return 2 * index;
    } else {
        return 2 * index + 1;
    }
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



