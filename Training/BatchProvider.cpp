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
   const bool is_v2 = (in_format == InputFormat::V2);
   //needs some refactoring at some point
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
/* 
    auto loop_condition =[&](Sample&current)->bool{
        if(is_v1){
            return (current.result == UNKNOWN || (current.position.has_jumps()) || current.move == -1);
        }else{
            return (current.result == UNKNOWN  || current.move == -1);
        }
    };
 */

    for (auto i = 0; i < get_batch_size(); ++i) {
        Sample current;
        do {
            current = get_streamer().get_next();
        } while (current.result == UNKNOWN || current.position.has_jumps());
        size_t off = INPUT_SIZE * i;
        auto result = create_input(current, inputs, off);
        results[i] = result;
        moves[i] = current.move;
    }


}
/* 
void BatchProvider::next(float *results, int64_t *moves, float *inputs) {
    static constexpr size_t INPUT_SIZE = 120;
    auto create_input = [](Sample s, float *input, size_t off) {
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
        Sample current;
        do {
            current = get_streamer().get_next();
        } while (current.result == UNKNOWN);
        size_t off = INPUT_SIZE * i;
        auto result = create_input(current, inputs, off);
        results[i] = result;
        moves[i] = current.move;
    }


}
 */

void BatchProvider::next_pattern(float*results,float* mover,int64_t* op_pawn_index,int64_t* end_pawn_index,int64_t* op_king_index,int64_t* end_king_index,float* wk_input,float*bk_input,float* wp_input,float*bp_input){
 
auto create_pattern_input =[&](Sample s, float*mover,int64_t* op_pawn_index,int64_t* end_pawn_index,int64_t* op_king_index,int64_t* end_king_index,float* wk_input,float*bk_input,float* wp_input,float*bp_input, size_t off,size_t pawn_off,size_t king_off){

        float result = 0.5f;
        if (s.result == BLACK_WON) {
            result = 0.0f;
        } else if (s.result == WHITE_WON) {
            result = 1.0f;
        }

        Position copy = s.position.get_color_flip();;
        size_t counter_op =0;
        size_t counter_end =0;
     
        

         Bits::big_index([&](size_t index){
            op_pawn_index[counter_op++ +pawn_off]=index;
            end_pawn_index[counter_end++ +pawn_off]=index+1;
        },copy.WP,copy.BP,copy.K);
        counter_op =0;
        counter_end =0;
        Bits::small_index([&](size_t index){
            op_king_index[counter_op++ +king_off]=index;
            end_king_index[counter_end++ +king_off]=index+1;
        },copy.WP,copy.BP,copy.K); 

   
        auto wk = Bits::pop_count(s.position.get_pieces<WHITE,KING>());
        auto bk =  Bits::pop_count(s.position.get_pieces<BLACK,KING>());
        auto wp = Bits::pop_count(s.position.get_pieces<WHITE,PAWN>());
        auto bp =  Bits::pop_count(s.position.get_pieces<BLACK,PAWN>());
        wk_input[off]=wk;
        bk_input[off]=bk;
        wp_input[off]=wp;
        bp_input[off]=bp;
        mover[off] = s.position.get_color();
        return result;
    };
    size_t pawn_off =6;
    size_t king_off = 9;
   for (auto i = 0; i < get_batch_size(); ++i) {
        Sample current;

        do {
            current = get_streamer().get_next();
        } while (current.result == UNKNOWN || (current.position.has_jumps()));

        auto result = create_pattern_input(current,mover, op_pawn_index,end_pawn_index,op_king_index,end_king_index,wk_input,bk_input,wp_input,bp_input,i,pawn_off*i,king_off*i);
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

