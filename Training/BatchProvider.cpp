//
// Created by robin on 07.10.21.
//

#include "BatchProvider.h"

void BatchProvider::set_input_format(InputFormat format){
    in_format = format;
    streamer.set_input_format(in_format);
}

void BatchProvider::next(float *results, int64_t *moves, float *inputs) {
   const bool is_v1 = (in_format == InputFormat::V1);
    auto create_input = [&](Sample s, float *input, size_t off) {
        if (s.position.color == BLACK) {
            s.position = s.position.get_color_flip();
            s.result = ~s.result;
        }
        float result = 0.5f;
        if (s.result == BLACK_WON) {
            result = 0.0f;
        } else if (s.result == WHITE_WON) {
            result = 1.0f;
        }


        uint32_t white_men = s.position.WP & (~s.position.K);
        uint32_t black_men = s.position.BP & (~s.position.K);
        uint32_t white_kings = s.position.K & s.position.WP;
        uint32_t black_kings = s.position.K & s.position.BP;
        size_t offset = 0u + off;

        while (white_men != 0u) {
            auto index = Bits::bitscan_foward(white_men);
            white_men &= white_men - 1u;
            input[offset + index - 4*(is_v1)] = 1;
        }
        offset += (is_v1)?28:32;
        while (black_men != 0u) {
            auto index = Bits::bitscan_foward(black_men);
            black_men &= black_men - 1u;
            input[offset + index] = 1;
        }
        offset += (is_v1)?28:32;;
        while (white_kings != 0u) {
            auto index = Bits::bitscan_foward(white_kings);
            white_kings &= white_kings - 1u;
            input[offset + index] = 1;
        }
        offset +=32;
        while (black_kings != 0u) {
            auto index = Bits::bitscan_foward(black_kings);
            black_kings &= black_kings - 1u;
            input[offset + index] = 1;
        }
        return result;
    };
    const size_t INPUT_SIZE = (in_format == InputFormat::V1)?120 : 128;

    auto loop_condition =[&](Sample&current)->bool{
        if(is_v1){
            return (current.result == UNKNOWN || (current.position.has_jumps()) || current.move == -1);
        }else{
            return (current.result == UNKNOWN  || current.move == -1);
        }
    };


    for (auto i = 0; i < get_batch_size(); ++i) {
        Sample current;
        do {
            current = get_streamer().get_next();
        } while (current.position.is_empty() || loop_condition(current));
        size_t off = INPUT_SIZE * i;
        auto result = create_input(current, inputs, off);
        results[i] = result;
        moves[i] = current.move;
    }


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

