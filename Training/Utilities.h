#ifndef UTILS
#define UTILS
#include "Network.h"
#include <fstream>
#include <string>
#include <iostream>
#include "GameLogic.h"
#include "Simd.h"
#include <filesystem>
namespace Utils{

    template <typename Net>
    std::vector<size_t> compute_histogramm(Net &net, std::string data_path)
    {
        constexpr auto num_activations = network.accumulator.out_dimension / 2;
        constexpr auto num_blocks = num_activations / 4;
        std::vector<size_t> histogram(num_activations);
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

            int32_t *acc_out = reinterpret_cast<int32_t *>(network.input);

            for (auto i = 0; i < num_activations; ++i)
            {
                histogram[i] += (network.input[i] != 0);
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

    // TODO check if type t is integral
    template <typename T>
    void save_vector_data(std::vector<T> data, std::string path)
    {
        std::ofstream stream(path, std::ios::binary);
        stream.write((char *)&data[0], sizeof(T) * data.size());
    }

    template <typename T>
    std::vector<T> read_vector_data(std::string path)
    {
        std::filesystem::path p(path);
        const auto file_size = std::filesystem::file_size(p);
        const auto num_items = file_size / sizeof(T);
        std::ifstream stream(path, std::ios::binary);

        std::vector<T> result(num_items);
        stream.read((char *)&result[0], file_size);
    }

    //returns a permutation which will hopefully improve sparsity
    //very simple idea -> sorting the activations based on a histogram
    std::vector<size_t> get_permutation_var1(std::vector<size_t> histogram, std::string out_path);
    std::vector<size_t> get_identity(std::vector<size_t> histogram);
}
#endif