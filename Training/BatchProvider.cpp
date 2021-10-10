//
// Created by robin on 07.10.21.
//

#include "BatchProvider.h"


void BatchProvider::next(float *results, float *inputs) {
    static constexpr size_t INPUT_SIZE = 120;
    auto create_input = [](Sample s, float *input, size_t off) {
        if (s.position.color == BLACK) {
            s.position = s.position.getColorFlip();
            s.result = -s.result;
        }
        float result;
        /*      res_temp = 1
              if res == 1 else 0
              if res == -1 else 0.5;*/
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

    for (auto i = 0; i < batch_size; ++i) {
        Sample current;
        do {
            current = streamer.get_next();
        } while (current.position.hasJumps(current.position.getColor()));
        size_t off = INPUT_SIZE * i;
        auto result = create_input(current, inputs, off);
        results[i] = result;
    }


}

size_t BatchProvider::get_batch_size() const {
    return batch_size;
}

size_t BatchProvider::get_buffer_size() const {
    return buffer_size;
}

const PosStreamer &BatchProvider::get_streamer() const {
    return streamer;
}
