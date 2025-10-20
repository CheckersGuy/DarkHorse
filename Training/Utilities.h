#ifndef UTILS
#define UTILS
#include "Network.h"
#include <fstream>
#include <string>
#include <iostream>
#include "GameLogic.h"
#include "Simd.h"
namespace Utils{

    template <typename Net>
    std::vector<size_t> compute_histogramm(Net &net, std::string data_path)
    {
        constexpr auto num_activations = network.accumulator.out_dimension / 2;
        constexpr auto num_blocks = num_activations / 4;
        alignas(64) uint8_t output_buffer[num_activations * 2] = {0};
        std::vector<size_t> histogram(num_activations * 2,0);
        std::ifstream stream(data_path);
        if (!stream.good())
        {
            std::cout << "Could not load stream" << std::endl;
            return histogram;
        }
        std::string fen_string;
        size_t num_active = 0;
        size_t num_samples = 0;
        size_t block_num_active = 0;
        size_t block_num_samples = 0;


        while (std::getline(stream, fen_string))
        {
            if (fen_string.empty()){
                continue;
            }
            const auto pos = Position::pos_from_fen(fen_string);

            int16_t *accu_buff;

            network.accumulator.forward(network.input, pos);

            if(pos.color == BLACK){
                accu_buff = network.accumulator.black_acc;
            }else{
                accu_buff = network.accumulator.white_acc;
            }

            Simd::accum_clipping8<2*num_activations>(accu_buff, &output_buffer[0]);

            int32_t *acc_out = reinterpret_cast<int32_t *>(network.input);

            for (auto i = 0; i < num_activations * 2; ++i)
            {
                histogram[i] += (output_buffer[i] != 0);
            }

            for (auto i = 0; i < network.accumulator.out_dimension / 2; ++i)
            {
                num_active += (network.input[i] != 0);
            }

            for (auto i = 0; i < network.accumulator.out_dimension / 8; ++i)
            {
                block_num_active += (acc_out[i] != 0);
            }
            num_samples++;
        }
        //computing the average

        const double average = static_cast<double>(num_active) / static_cast<double>(num_activations * num_samples);
        const double block_average = static_cast<double>(block_num_active) / static_cast<double>(num_blocks* num_samples);
        std::cout << "Average: " << average << std::endl;
        std::cout << "Average_Block: " << block_average << std::endl;
        return histogram;
    }


    //returns a permutation which will hopefully improve sparsity
    //very simple idea -> sorting the activations based on a histogram
    std::vector<size_t> get_permutation_var1(std::vector<size_t> histogram, std::string out_path);
    std::vector<size_t> get_identity(std::vector<size_t> histogram);
}
#endif