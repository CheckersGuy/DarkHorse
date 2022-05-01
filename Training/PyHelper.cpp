//
// Created by root on 06.01.21.
//

#include "PyHelper.h"
#include <PosStreamer.h>
#include <BatchProvider.h>

std::unique_ptr<BatchProvider> streamer;
std::unique_ptr<BatchProvider> val_streamer;
extern "C" int
init_streamer(size_t buffer_size, size_t batch_size, size_t a, size_t b, char *file_path, bool patterns) {
    std::string path(file_path);
    std::cout << "Path: " << file_path << std::endl;
    if (streamer.get() == nullptr) {
            streamer = std::make_unique<NetBatchProvider>(path, buffer_size, batch_size, a, b);
    }
    std::cout<<"NumPositions in training set: "<<streamer->get_streamer().get_num_positions()<<std::endl;
    return streamer->get_streamer().get_num_positions();
}

extern "C" int
init_val_streamer(size_t buffer_size, size_t batch_size, size_t a, size_t b, char *file_path, bool patterns) {
    std::string path(file_path);
    std::cout << "Path: " << file_path << std::endl;
    if (val_streamer.get() == nullptr) {
        val_streamer = std::make_unique<NetBatchProvider>(path, buffer_size, batch_size, a, b);
    }
    std::cout<<"NumPositions in validation set: "<<val_streamer->get_streamer().get_num_positions()<<std::endl;
    return val_streamer->get_streamer().get_num_positions();
}

extern "C" int
init_streamer2(size_t buffer_size, size_t batch_size, size_t a, size_t b, char *file_path, bool patterns) {
    std::string path(file_path);
    std::cout << "Path: " << file_path << std::endl;
    if (streamer.get() == nullptr) {
            streamer = std::make_unique<NetBatchProvider2>(path, buffer_size, batch_size, a, b);
    }
    std::cout<<"NumPositions in training set: "<<streamer->get_streamer().get_num_positions()<<std::endl;
    return streamer->get_streamer().get_num_positions();
}

extern "C" int
init_val_streamer2(size_t buffer_size, size_t batch_size, size_t a, size_t b, char *file_path, bool patterns) {
    std::string path(file_path);
    std::cout << "Path: " << file_path << std::endl;
    if (val_streamer.get() == nullptr) {
        val_streamer = std::make_unique<NetBatchProvider2>(path, buffer_size, batch_size, a, b);
    }
    std::cout<<"NumPositions in validation set: "<<val_streamer->get_streamer().get_num_positions()<<std::endl;
    return val_streamer->get_streamer().get_num_positions();
}


extern "C" void get_next_batch(float *results, int64_t *moves, float *inputs) {
    if (streamer.get() == nullptr) {
        std::exit(-1);
    }
    NetBatchProvider *provider = static_cast<NetBatchProvider *>(streamer.get());
    provider->next(results, moves, inputs);
}

extern "C" void get_next_val_batch(float *results, int64_t *moves, float *inputs) {
    if (val_streamer.get() == nullptr) {
        std::exit(-1);
    }
    NetBatchProvider *provider = static_cast<NetBatchProvider *>(val_streamer.get());
    provider->next(results, moves, inputs);
}


extern "C" void get_next_batch2(float *results, int64_t *moves, float *inputs) {
    if (streamer.get() == nullptr) {
        std::exit(-1);
    }
    NetBatchProvider2 *provider = static_cast<NetBatchProvider2 *>(streamer.get());
    provider->next(results, moves, inputs);
}

extern "C" void get_next_val_batch2(float *results, int64_t *moves, float *inputs) {
    if (val_streamer.get() == nullptr) {
        std::exit(-1);
    }
    NetBatchProvider2 *provider = static_cast<NetBatchProvider2 *>(val_streamer.get());
    provider->next(results, moves, inputs);
}


extern "C" void print_fen(const char* fen_string) {
    Position pos = Position::pos_from_fen(std::string(fen_string));
    pos.print_position();
}

extern "C" void print_position(uint32_t white_men, uint32_t black_men, uint32_t kings) {
    Position pos;
    pos.WP = white_men;
    pos.BP = black_men;
    pos.K = kings;
    pos.print_position();
}

extern "C" void get_input_from_fen(float* inputs,const char* fen_string){
    const Position pos = Position::pos_from_fen(std::string(fen_string));
     auto create_input=[&](Sample s,float*input,size_t offset){
        if(s.position.get_color()==BLACK){
            s.position = s.position.get_color_flip();
            s.result=~s.result;
        }
        float result = 0.5f;
        if (s.result == BLACK_WON) {
            result = 0.0f;
        } else if (s.result == WHITE_WON) {
            result = 1.0f;
        }

        float* in_plane =input +offset;
        //to be continued

        uint32_t white_men = s.position.WP & (~s.position.K);
        uint32_t black_men = s.position.BP & (~s.position.K);
        uint32_t white_kings = s.position.K & s.position.WP;
        uint32_t black_kings = s.position.K & s.position.BP;


        while (white_men != 0u) {
            auto index = Bits::bitscan_foward(white_men);
            white_men &= white_men - 1u;
            input[offset + index] = 1;
        }
        offset += 32;
        while (black_men != 0u) {
            auto index = Bits::bitscan_foward(black_men);
            black_men &= black_men - 1u;
            input[offset + index] = 1;
        }
        offset += 32;
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
    Sample dummy;
    dummy.position=pos;
    create_input(dummy,inputs,0);

}
