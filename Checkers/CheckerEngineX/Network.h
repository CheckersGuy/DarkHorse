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


class Network;

struct Layer {
    int in_features;
    int out_features;
};

struct Accumulator {
    Network& network;
    std::unique_ptr<float[]> black_acc;
    std::unique_ptr<float[]> white_acc;
    size_t size;
    

    Accumulator(size_t s,Network& network): network(network) {
        black_acc = std::make_unique<float[]>(s);
        white_acc= std::make_unique<float[]>(s);
        size = s;
    }



    Accumulator(const Accumulator &other): network(other.network){
    size = other.size;
    black_acc = std::make_unique<float[]>(size);
    white_acc = std::make_unique<float[]>(size);

    std::copy(other.black_acc.get(), other.black_acc.get() + size, black_acc.get());
    std::copy(other.white_acc.get(), other.white_acc.get() + size, white_acc.get());
}

    Accumulator& operator=(const Accumulator& other);
    Accumulator() = default;

    //to be implemented
    void update(Color per, Position before,Position after);

    void add_feature(Color perp,size_t index);

    void add_feature(Color perp,Position& before, Position&after);

    void remove_feature(Color perp,size_t index);

    void remove_feature(Color perp,Position& before, Position&after);
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

    friend class Accumulator;
};


#endif //READING_NETWORK_H
