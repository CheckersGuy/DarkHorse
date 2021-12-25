//
// Created by root on 06.01.21.
//

#include "PyHelper.h"
#include <PosStreamer.h>
#include <BatchProvider.h>

std::unique_ptr<BatchProvider> streamer;
std::unique_ptr<BatchProvider> val_streamer;
extern "C" int init_streamer(size_t buffer_size, size_t batch_size, char *file_path, bool patterns) {
    std::string path(file_path);
    std::cout << "Path: " << file_path << std::endl;
    if (streamer.get() == nullptr) {
        if (!patterns) {
            streamer = std::make_unique<NetBatchProvider>(path, buffer_size, batch_size);
        } else {
            streamer = std::make_unique<PattBatchProvider>(path, buffer_size, batch_size);
        }

    }
    return streamer->get_streamer().get_num_positions();
}

extern "C" int init_val_streamer(size_t buffer_size, size_t batch_size, char *file_path, bool patterns) {
    std::string path(file_path);
    std::cout << "Path: " << file_path << std::endl;
    if (val_streamer.get() == nullptr) {
        if (!patterns) {
            val_streamer = std::make_unique<NetBatchProvider>(path, buffer_size, batch_size);
        } else {
            val_streamer = std::make_unique<PattBatchProvider>(path, buffer_size, batch_size);
        }
    }
    return val_streamer->get_streamer().get_num_positions();
}


extern "C" void get_next_batch(float *results,int64_t *moves, float *inputs) {
    if (streamer.get() == nullptr) {
        std::exit(-1);
    }
    NetBatchProvider *provider = static_cast<NetBatchProvider *>(streamer.get());
    provider->next(results,moves, inputs);
}

extern "C" void get_next_val_batch(float *results,int64_t *moves, float *inputs) {
    if (val_streamer.get() == nullptr) {
        std::exit(-1);
    }
    NetBatchProvider *provider = static_cast<NetBatchProvider *>(val_streamer.get());
    provider->next(results,moves, inputs);
}

extern "C" void
get_next_batch_patt(float *results, float *num_wp, float *num_bp, float *num_wk, float *num_bk, int64_t *patt_op_big,
                    int64_t *patt_end_big, int64_t *patt_op_small,
                    int64_t *patt_end_small) {
    if (streamer.get() == nullptr) {
        std::exit(-1);
    }
    PattBatchProvider *provider = static_cast<PattBatchProvider *>(streamer.get());
    provider->next(results, num_wp, num_bp, num_wk, num_bk, patt_op_big,
                   patt_end_big, patt_op_small,
                   patt_end_small);
}

extern "C" void get_next_val_batch_patt(float *results, float *num_wp, float *num_bp, float *num_wk, float *num_bk, int64_t *patt_op_big,
                                        int64_t *patt_end_big, int64_t *patt_op_small,
                                        int64_t *patt_end_small) {
    if (val_streamer.get() == nullptr) {
        std::exit(-1);
    }
    PattBatchProvider *provider = static_cast<PattBatchProvider *>(val_streamer.get());
    provider->next(results, num_wp, num_bp, num_wk, num_bk, patt_op_big,
                   patt_end_big, patt_op_small,
                   patt_end_small);
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



