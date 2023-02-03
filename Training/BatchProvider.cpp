//
// Created by robin on 07.10.21.
//

#include "BatchProvider.h"
#include "generator.pb.h"



void BatchProvider::next(float *results, int64_t *moves, float *inputs) {
    //needs some refactoring at some point
    auto create_input = [&](Proto::Sample s, float *input, size_t off) {
        if (s.mover() == Proto::BLACK) {
            Position temp;
            temp.WP = s.wp();
            temp.BP =s.bp();
            temp.K =s.k();
            temp = temp.get_color_flip();

            s.set_wp(temp.WP);
            s.set_bp(temp.BP);
            s.set_k(temp.K);
            s.set_mover(Proto::WHITE);
            s.set_result((s.result()==Proto::WHITE_WIN)?Proto::BLACK_WIN : Proto::WHITE_WIN);
        }
        float result = 0.5f;
        if (s.result()== Proto::BLACK_WIN) {
            result = 0.0f;
        } else if (s.result() == Proto::WHITE_WIN) {
            result = 1.0f;
        }


        uint32_t white_men = s.wp() & (~s.k());
        uint32_t black_men = s.bp() & (~s.k());
        uint32_t white_kings = s.k() & s.wp();
        uint32_t black_kings = s.k() & s.bp();
        size_t offset = 0u + off;

        while (white_men != 0u) {
            auto index = Bits::bitscan_foward(white_men)-4;
            white_men &= white_men - 1u;
            input[offset + index] = 1;
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
        offset +=32;
        while (black_kings != 0u) {
            auto index = Bits::bitscan_foward(black_kings);
            black_kings &= black_kings - 1u;
            input[offset + index] = 1;
        }
        return result;
    };



    const size_t INPUT_SIZE = 120;
    for (auto i = 0; i < get_batch_size(); ++i) {
        Proto::Sample current = get_streamer().get_next();
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

PosStreamer &BatchProvider::get_streamer() {
    return streamer;
}

