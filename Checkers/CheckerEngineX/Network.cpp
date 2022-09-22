//
// Created by root on 18.04.21.
//

#include <complex>
#include "Network.h"

    void Accumulator::refresh(){
        for(auto i=0;i<size;++i){
            white_acc[i]=net->biases[i];
            black_acc[i]=net->biases[i];
        }
        previous_black = Position{};
        previous_white =Position{};
    }

    void Accumulator::apply(Color perp, Position before, Position after)
    {
        int16_t* input =((perp == BLACK)?black_acc.get(): white_acc.get());

        auto WP_O = after.get_pieces<WHITE,PAWN>() & (~before.get_pieces<WHITE,PAWN>());
        auto BP_O = after.get_pieces<BLACK,PAWN>() & (~before.get_pieces<BLACK,PAWN>());
        auto WK_O = after.get_pieces<WHITE,KING>() & (~before.get_pieces<WHITE,KING>());
        auto BK_O = after.get_pieces<BLACK,KING>() & (~before.get_pieces<BLACK,KING>());

        size_t offset =0;


        while (WP_O)
        {
            auto index = Bits::bitscan_foward(WP_O) - 4+offset;
            add_feature(input, index);
            WP_O &= WP_O - 1;
        }
        offset += 28;

        while (BP_O)
        {
            auto index = Bits::bitscan_foward(BP_O)+offset;
            add_feature(input, index);
            BP_O &= BP_O - 1;
        }
        offset += 28;

        while (WK_O)
        {
            auto index = Bits::bitscan_foward(WK_O)+offset;
            add_feature(input, index);
            WK_O &= WK_O - 1;
        }
        offset += 32;

        while (BK_O)
        {
            auto index = Bits::bitscan_foward(BK_O)+offset;
            add_feature(input, index);
            BK_O &= BK_O - 1;
        } 

        auto WP_Z = (~after.get_pieces<WHITE,PAWN>()) & (before.get_pieces<WHITE,PAWN>());
        auto BP_Z = (~after.get_pieces<BLACK,PAWN>()) & (before.get_pieces<BLACK,PAWN>());
        auto WK_Z = (~after.get_pieces<WHITE,KING>()) & (before.get_pieces<WHITE,KING>());
        auto BK_Z = (~after.get_pieces<BLACK,KING>()) & (before.get_pieces<BLACK,KING>());

        offset =0;

        // to be continued
        while (WP_Z)
        {
            auto index = Bits::bitscan_foward(WP_Z) - 4+offset;
            remove_feature(input, index);
            WP_Z &= WP_Z - 1;
        }
        offset += 28;

        while (BP_Z)
        {
            auto index = Bits::bitscan_foward(BP_Z)+offset;
            remove_feature(input, index);
            BP_Z &= BP_Z - 1;
        }
        offset += 28;

        while (WK_Z)
        {
            auto index = Bits::bitscan_foward(WK_Z)+offset;
            remove_feature(input, index);
            WK_Z &= WK_Z - 1;
        }
        offset += 32;

        while (BK_Z)
        {
            auto index = Bits::bitscan_foward(BK_Z)+offset;
            remove_feature(input, index);
            BK_Z &= BK_Z - 1;
        }

    }

void Accumulator::update(Color perp,Position after) {
    if (perp == BLACK){
        apply(perp, previous_black.get_color_flip(), after.get_color_flip());
        previous_black = after;
    }
    else{
        apply(perp, previous_white, after);
        previous_white = after;
    }
    
}

void Accumulator::init(Network* net){
    const auto size = net->layers[0].out_features;
    black_acc = std::make_unique<int16_t[]>(size);
    white_acc = std::make_unique<int16_t[]>(size);
    this->size =size;
    this->net = net;

      for(auto i=0;i<this->net->layers[0].out_features;++i){
        black_acc[i]=net->biases[i];
        white_acc[i]=net->biases[i];
    }

}

void Accumulator::add_feature(int16_t * input,size_t index){
    //adding the index-th column to our feature vector
    for(auto i=0;i<size;++i){
        input[i] += net->weights[index*net->layers[0].out_features+i];
    }
}

void Accumulator::remove_feature(int16_t* input,size_t index){
    //adding the index-th column to our feature vector
    for(auto i=0;i<size;++i){
        input[i] -= net->weights[index*net->layers[0].out_features+i];
    }
}

void Network::load(std::string file) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream.good())
    {
        std::cerr << "Could not load the weights" << std::endl;
        std::exit(-1);
    }

    //experimental bullshit
/* 
    int num_layers;
    for(auto i=0;i<num_layers;++i){
        int layer_size;
        stream.read((char*)&layer_size,sizeof(int));
    } */

    int num_weights, num_bias;
    stream.read((char *)&num_weights, sizeof(int));
    weights = std::make_unique<int16_t[]>(num_weights);
    stream.read((char *)weights.get(), sizeof(int16_t) * num_weights);
    stream.read((char *)&num_bias, sizeof(int));
    biases = std::make_unique<int16_t[]>(num_bias);
    stream.read((char *)biases.get(), sizeof(int16_t) * num_bias);
    stream.close();
}

void Network::addLayer(Layer layer) {
    layers.emplace_back(layer);
}

void Network::init() {
    for (Layer l : layers)
    {
        max_units = std::max(std::max(l.in_features, l.out_features), max_units);
    }

    temp = std::make_unique<int16_t[]>(max_units);
    input = std::make_unique<int16_t[]>(max_units);

    for (auto i = 0; i < max_units; ++i) {
        temp[i] = 0;
        input[i] = 0;
    }

    accumulator.init(this);
  
}

int16_t Network::get_max_weight() const {
    auto max_value = std::numeric_limits<int16_t>::min();
    size_t num_weights = 0;
    for (Layer l: layers) {
        num_weights += l.out_features * l.in_features;
    }
    for (int i = 0; i < num_weights; ++i) {
          if(std::abs(weights[i])>max_value){
            max_value = std::abs(weights[i]);
        }
    }

    return max_value;

}

int16_t Network::get_max_bias() const {
    auto max_value = std::numeric_limits<int16_t>::min();
    size_t num_bias= 0;
    for (Layer l: layers) {
        num_bias += l.out_features;
    }
    for (int i = 0; i < num_bias; ++i) {
        if(std::abs(biases[i])>max_value){
            max_value = std::abs(biases[i]);
        }
    }

    return max_value;

}

int16_t Network::compute_incre_forward_pass(Position next) {
    int16_t *z_previous;
 
    if (next.color == BLACK) {
        z_previous = accumulator.black_acc.get();
    } else {
        z_previous = accumulator.white_acc.get();
    }
    accumulator.update(next.color,next);
    for (auto i = 0; i < layers[0].out_features; i++) {
        temp[i] = std::clamp(z_previous[i], int16_t{0}, int16_t{127});
    }

    for (auto i = 0; i < layers[0].out_features; ++i)
    {
        input[i] = temp[i];
        temp[i] = 0;
    }
    auto weight_index_offset = layers[0].out_features*layers[0].in_features;;
    auto bias_index_offset = layers[0].out_features;
    //computation for the remaining layers
    for (auto k = 1; k < layers.size(); ++k) {
        Layer l = layers[k];

        for (auto i = 0; i < l.out_features; i++)
        {
            int sum =biases[i+bias_index_offset];
            for (auto j = 0; j < l.in_features; ++j)
            {
                sum+= weights[weight_index_offset + i * l.in_features + j] * input[j];
            }
             if (k < layers.size() - 1) {
                temp[i] = std::clamp(sum/64,  0, 127);
            }else{
                return sum/64; 
            
            }
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

int Network::operator[](size_t index){
    return input[index];
}

int Network::evaluate(Position pos, int ply)
{

    if (pos.BP == 0)
    {
        return -loss(ply);
    }
    if (pos.WP == 0)
    {
        return loss(ply);
    }

    int16_t val = compute_incre_forward_pass(pos) / 2;
    return val;
}

int Network::evaluate(Position pos, int ply, Network &net1, Network &net2) {
    auto num_pieces = Bits::pop_count(pos.BP | pos.WP);
    if (num_pieces>12) {
        return net1.evaluate(pos, ply);
    } else {
        return net2.evaluate(pos, ply);
    }
}



