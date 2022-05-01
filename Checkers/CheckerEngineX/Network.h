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

struct Layer {
    int in_features;
    int out_features;
};

inline double sigmoid(double d){
    return 1.0/(1.0 + std::exp(-d));
}

struct Accumulator {
    std::unique_ptr<float[]> acc;

    Accumulator(size_t size) {
        acc = std::make_unique<float[]>(2 * size);
    }

    Accumulator() = default;

    //to be implemented
    void update(Color perp, uint32_t a_WP, uint32_t a_BP, uint32_t a_WK, uint32_t a_BK, uint32_t r_WP, uint32_t r_BP,
                uint32_t r_WK, uint32_t r_BK);
};

std::pair<uint32_t, uint32_t> compute_difference(uint32_t previous, uint32_t next);


struct Network {

    Position p_black, p_white;
    std::vector<Layer> layers;
    std::unique_ptr<float[]> biases;
    std::unique_ptr<float[]> weights;
    std::unique_ptr<float[]> input;
    std::unique_ptr<float[]> temp;
    std::unique_ptr<float[]> z_black;
    std::unique_ptr<float[]> z_white;
    int max_units{0};


    float get_max_weight();

    void addLayer(Layer layer);

    void load(std::string file);

    void init();

    void set_input(Position p);

    float compute_incre_forward_pass(Position next);

    float *get_output();

    float forward_pass() const;

    float get_win_p(Position pos);

    int evaluate(Position pos, int ply);

    static int evaluate(Position pos, int ply, Network& net1, Network& net2);

};


#endif //READING_NETWORK_H
