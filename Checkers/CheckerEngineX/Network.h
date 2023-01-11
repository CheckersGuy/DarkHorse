//
// Created by root on 18.04.21.
//

#ifndef READING_NETWORK_H
#define READING_NETWORK_H

#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include "Position.h"
#include <cstring>
#include <immintrin.h>


class Network;

struct Layer {
    int in_features;
    int out_features;
};

struct Accumulator {
    std::unique_ptr<int16_t[]> black_acc;
    std::unique_ptr<int16_t[]> white_acc;
    size_t size;
    Position previous_black,previous_white;
    Network* net = nullptr;

    void init(Network* net);

    void update(Color per,Position after);

    void add_feature(int16_t* input,size_t index);

    void remove_feature(int16_t* input,size_t index);

    void apply(Color color,Position before, Position after);

    void refresh();

};

template<typename T> void display_network_data(std::string network_file){
        std::ifstream stream(network_file, std::ios::binary);
    if (!stream.good()) {
        std::cerr << "Could not load the weights" << std::endl;
        std::exit(-1);
    }




    int num_weights, num_bias;
    stream.read((char *) &num_weights, sizeof(int));
    auto weights = std::make_unique<T[]>(num_weights);

    stream.read((char *) weights.get(), sizeof(T) * num_weights);
    stream.read((char *) &num_bias, sizeof(int));
    auto biases = std::make_unique<T[]>(num_bias);
    stream.read((char *) biases.get(), sizeof(T) * num_bias);

     for (auto i = 0; i < num_bias; ++i)
    {
        std::cout << biases[i] << std::endl;
    } 

    stream.close();
}



struct Network {
    std::vector<Layer> layers;
    std::unique_ptr<int16_t[]> ft_biases;
    std::unique_ptr<int16_t[]> ft_weights;
    std::unique_ptr<int32_t[]> biases;
    std::unique_ptr<int16_t[]> weights;
    std::unique_ptr<int16_t[]> input;
    std::unique_ptr<int16_t[]> temp;
    int max_units{0};
    Accumulator accumulator;


    int32_t get_max_weight() const;

    int32_t get_max_bias() const;

    void addLayer(Layer layer);

    void load(std::string file);

    void init();

    void print_output_layer();

    int compute_incre_forward_pass(Position next);

    float get_win_p(Position pos);

    int evaluate(Position pos, int ply);

    static int evaluate(Position pos, int ply, Network& net1, Network& net2);
    
    int operator[](size_t index);

    friend class Accumulator;
};

void testing_simd_functions();
#endif //READING_NETWORK_H
