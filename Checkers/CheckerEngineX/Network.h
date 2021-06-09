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
#include "SIMD.h"
#include <cstring>

float sigmoid(float val);


struct Layer {
    int in_features;
    int out_features;
};

std::pair<uint32_t, uint32_t> compute_difference(uint32_t previous, uint32_t next);

struct AlignedDeleter {
    void operator()(float *ptr) {
        //needs to be different fo MSVC
        _mm_free(ptr);
    }
};

template<typename T> using aligned_ptr = std::unique_ptr<T,AlignedDeleter>;

struct Network {

    Position p_black, p_white;
    std::vector<Layer> layers;
    aligned_ptr<float[]> biases;
    aligned_ptr<float[]> weights;
    aligned_ptr<float[]> input;
    aligned_ptr<float[]> temp;
    aligned_ptr<float[]> z_black;
    aligned_ptr<float[]> z_white;
    int max_units{0};


    float get_max_weight();

    void addLayer(Layer layer);

    void load(std::string file);

    void init();

    void set_input(Position p);

    float compute_incre_forward_pass(Position next);

    float forward_pass();

    int evaluate(Position pos);

};


#endif //READING_NETWORK_H
