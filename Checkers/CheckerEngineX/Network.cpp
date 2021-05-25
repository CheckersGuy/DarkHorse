//
// Created by root on 18.04.21.
//

#include <complex>
#include "Network.h"

float sigmoid(float value) {
    return 1.0f / (1.0f + std::exp(-value));
}

std::pair<uint32_t, uint32_t> compute_difference(uint32_t previous, uint32_t next) {
    //computes the indices which need to be either set to 0
    //or for which we need to compute the matrix-vec multiply
    uint32_t changed_to_ones = next & (~previous);
    uint32_t n_previous = ~previous;
    uint32_t n_next = ~next;
    uint32_t changed_to_zeros = n_next&(~n_previous);
    return std::pair<uint32_t, uint32_t>(changed_to_zeros,changed_to_ones);
}

void Network::load(std::string file) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream.good())
        std::exit(-1);
    uint32_t num_weights, num_bias;
    stream.read((char *) &num_weights, sizeof(uint32_t));
    weights = std::make_unique<float[]>(num_weights);
    stream.read((char *) weights.get(), sizeof(float) * num_weights);
    stream.read((char *) &num_bias, sizeof(float));
    biases = std::make_unique<float[]>(num_bias);
    stream.read((char *) biases.get(), sizeof(float) * num_bias);
    stream.close();

}

void Network::addLayer(Layer layer) {
    layers.emplace_back(layer);
}

void Network::init() {
    for (Layer l : layers)
        max_units = std::max(std::max(l.in_features, l.out_features), max_units);

    temp = std::make_unique<float[]>(max_units);
    input = std::make_unique<float[]>(max_units);

    for (auto i = 0; i < max_units; ++i) {
        temp[i] = 0;
        input[i] = 0;
    }
}

float Network::forward_pass() {
    size_t weight_index_offset = 0u;
    size_t bias_index_offset = 0u;
    for (auto k = 0; k < layers.size(); ++k) {
        Layer &l = layers[k];
        for (auto j = 0; j < l.in_features; ++j) {
            if (input[j] == 0)
                continue;
            for (auto i = 0; i < l.out_features; ++i) {
                temp[i] += weights[weight_index_offset + j * l.out_features + i] * input[j];
            }
        }
        for (auto i = 0; i < l.out_features; ++i) {
            temp[i] += biases[bias_index_offset + i];
            if (k < layers.size() - 1)
                temp[i] = std::clamp(temp[i], 0.0f, temp[i]);
        }

        for (auto i = 0; i < l.out_features; ++i) {
            input[i] = temp[i];
            temp[i] = 0;
        }
        weight_index_offset += l.out_features * l.in_features;
        bias_index_offset += l.out_features;

    }
    return input[0];
}

int Network::evaluate(Position pos) {
    set_input(pos);
    float val = forward_pass();
    return static_cast<int>(val);
}

void Network::set_input(Position p) {
    //testing another network architecture
    if (p.color == BLACK) {
        p = p.getColorFlip();
    }
    constexpr size_t INPUT_SIZE = 120;
    //clearing the input first

    for (auto i = 0; i < max_units; ++i) {
        input[i] = 0;
        temp[i] = 0;
    }
    uint32_t white_men = p.WP & (~p.K);
    uint32_t black_men = p.BP & (~p.K);
    uint32_t white_kings = p.K & p.WP;
    uint32_t black_kings = p.K & p.BP;

    size_t offset = 0u;
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
    //input[INPUT_SIZE - 1] = (p.getColor() == BLACK) ? 0 : 1;
/*
    for (auto i = 0; i < INPUT_SIZE; ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
*/

}