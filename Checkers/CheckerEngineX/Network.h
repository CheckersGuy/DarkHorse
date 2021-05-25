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

float sigmoid(float val);




struct Layer {
    int in_features;
    int out_features;
};

std::pair<uint32_t ,uint32_t > compute_difference(uint32_t previous, uint32_t next);


class Network {
public:
    std::unique_ptr<float[]> biases;
    std::unique_ptr<float[]> weights;
    std::vector<Layer> layers;
    std::unique_ptr<float[]>input;
    std::unique_ptr<float[]>temp;
    int max_units{0};
public:

    void addLayer(Layer layer);

    void load(std::string file);

    void init();

    void set_input(Position p);

    float forward_pass();

    int evaluate(Position pos);

};


#endif //READING_NETWORK_H
