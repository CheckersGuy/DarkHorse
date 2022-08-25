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
    std::unique_ptr<float[]> black_acc;
    std::unique_ptr<float[]> white_acc;
    size_t size;
    Network* net = nullptr;

    void init(Network* net);

    void update(Color per, Position before,Position after);

    void add_feature(Color perp,size_t index);

    void add_feature(Color perp,Position before, Position after);

    void remove_feature(Color perp,size_t index);

    void remove_feature(Color perp,Position before, Position after);

    template <typename Func>
    void apply(Color perp, Position before, Position after, Func function)
    {
        auto WP = after.WP & (~before.WP);
        auto BP = after.BP & (~before.BP);
        auto WK = (after.WP & after.K) & (~(before.WP));
        auto BK = (after.BP & after.K) & (~(before.BP));

        size_t offset =0;

        // to be continued
        while (WP)
        {
            auto index = Bits::bitscan_foward(WP) - 4;
            function(perp, offset+index);
            WP &= WP - 1;
        }
        offset+=28;

        while (BP)
        {
            auto index = Bits::bitscan_foward(WP);
            function(perp, offset+index);
            BP &= BP - 1;
        }
        offset+=28;

        while (WK)
        {
            auto index = Bits::bitscan_foward(WK);
            function(perp, offset+index);
            WK &= WK - 1;
        }
        offset+=32;

        while (BK)
        {
            auto index = Bits::bitscan_foward(BK);
            function(perp, offset+index);
            BK &= BK - 1;
        }
    }
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
    Accumulator accumulator;


    float get_max_weight() const;

    float get_max_bias() const;

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
