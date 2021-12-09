//
// Created by robin on 07.10.21.
//

#include "BatchProvider.h"

void PattBatchProvider::next(float *results, float *num_wp, float *num_bp, float *num_wk, float *num_bk,
                             int64_t *patt_op_big,
                             int64_t *patt_end_big, int64_t *patt_op_small,
                             int64_t *patt_end_small) {
/*
    auto fill = [&](Sample s, size_t offset_small, size_t offset_big) {
        size_t big_counter = 0;
        size_t small_counter = 0;


        const size_t offset1 = 8ull * 157464ull;
        const size_t offset2 = 4ull * 531441ull + 8ull * 157464ull;
        if (s.position.K == 0) {
            //FOR THE PROMO_SQUARES

        } else {
            for (auto i = 0; i < 3; ++i) {
                for (auto k = 0; k < 3; ++k) {
                    const uint32_t sub_reg = region << (8 * i + k);
                    size_t index = getIndex2(sub_reg, s.position.WP, s.position.BP, s.position.K);
                    size_t sub_index_op = 18 * index + 2 * k + 6 * i;
                    size_t sub_index_end = 18 * index + 2 * k + 6 * i + 1;
                    patt_op_small[offset_small + small_counter_op++] = offset2 + sub_index_op;
                    patt_end_small[offset_small + small_counter_end++] = offset2 + sub_index_end;
                }
            }

        }
    };

    for (auto i = 0; i < get_batch_size(); ++i) {
        Sample s = get_streamer().get_next();
        size_t off_small = 9 * i;
        size_t off_big = 6 * i;

        num_wp[i] = (float) (Bits::pop_count(s.position.WP & (~s.position.K)));
        num_bp[i] = (float) (Bits::pop_count(s.position.BP & (~s.position.K)));
        num_wk[i] = (float) (Bits::pop_count(s.position.WP & s.position.K));
        num_bk[i] = (float) (Bits::pop_count(s.position.BP & s.position.K));
        if (s.position.getColor() == BLACK) {
            s.position = s.position.getColorFlip();
            s.result = -s.result;
        }
        float res_temp;
        if (s.result == -1)
            res_temp = 0;
        else if (s.result == 0)
            res_temp = 0.5;
        else
            res_temp = 1.0;
        results[i] = res_temp;

        fill(s, off_small, off_big);


    }*/

}


void NetBatchProvider::next(float *results, float *inputs) {
    static constexpr size_t INPUT_SIZE = 120;
    auto create_input = [](Sample s, float *input, size_t off) {
        if (s.position.color == BLACK) {
            s.position = s.position.getColorFlip();
            s.result = -s.result;
        }
        float result;
        if (s.result == -1)
            result = 0.0f;
        else if (s.result == 0)
            result = 0.5f;
        else if (s.result == 1)
            result = 1.0f;
        else
            result = 0.5f;
        uint32_t white_men = s.position.WP & (~s.position.K);
        uint32_t black_men = s.position.BP & (~s.position.K);
        uint32_t white_kings = s.position.K & s.position.WP;
        uint32_t black_kings = s.position.K & s.position.BP;


        size_t offset = 0u + off;
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
        return result;
    };

    for (auto i = 0; i < get_batch_size(); ++i) {
        Sample current = get_streamer().get_next();
        size_t off = INPUT_SIZE * i;
        auto result = create_input(current, inputs, off);
        results[i] = result;
    }


}

void NetBatchProvider2::next(float *results, float *patt1, float *patt2, float *patt3, float *patt4) {
    auto create_input = [](Sample s, float *patt1, float *patt2, float *patt3, float *patt4, size_t offset) {
        if (s.position.color == BLACK) {
            s.position = s.position.getColorFlip();
            s.result = -s.result;
        }
        float result;
        if (s.result == -1)
            result = 0.0f;
        else if (s.result == 0)
            result = 0.5f;
        else if (s.result == 1)
            result = 1.0f;
        else
            result = 0.5f;

        Position pos = s.position;
        //for now just a dummy below
        uint32_t reg = region;
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


        return result;
    };
}

size_t BatchProvider::get_batch_size() const {
    return batch_size;
}

size_t BatchProvider::get_buffer_size() const {
    return buffer_size;
}

PosStreamer &BatchProvider::get_streamer() {
    return streamer;
}
