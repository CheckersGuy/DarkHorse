//
// Created by root on 18.04.21.
//

#include "Network.h"
#include "GameLogic.h"
#include <cstdint>


void Accumulator::refresh() {
    for(auto i=0; i<size; ++i) {
        white_acc[i]=net->ft_biases[i];
        black_acc[i]=net->ft_biases[i];
    }
    previous_black = Position{};
    previous_white =Position{};
}


Wdl Network::get_wdl(){
 Wdl values;
 values.win = (*this)[0];
 values.loss = (*this)[1];
 values.draw =(*this)[2];
 return values;
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
    if (perp == BLACK) {
        apply(perp, previous_black.get_color_flip(), after.get_color_flip());
        previous_black = after;
    }
    else {
        apply(perp, previous_white, after);
        previous_white = after;
    }

}

void Accumulator::init(Network* net) {
    const auto size = net->layers[0].out_features;
    black_acc = std::make_unique<int16_t[]>(size);
    white_acc = std::make_unique<int16_t[]>(size);
    this->size =size;
    this->net = net;

    for(auto i=0; i<this->net->layers[0].out_features; ++i) {
        black_acc[i]=net->ft_biases[i];
        white_acc[i]=net->ft_biases[i];
    }

}

void Accumulator::add_feature(int16_t * input,size_t index) {
    //adding the index-th column to our feature vector
    for(auto i=0; i<size; ++i) {
        input[i] += net->ft_weights[index*net->layers[0].out_features+i];
    }
}

void Accumulator::remove_feature(int16_t* input,size_t index) {
    //adding the index-th column to our feature vector
    for(auto i=0; i<size; ++i) {
        input[i] -= net->ft_weights[index*net->layers[0].out_features+i];
    }
}


void Network::print_output_layer(){
  auto outputs = layers.back().out_features;

  for(int i=0;i<outputs;++i){
    std::cout<<"Index: "<<i<<" "<<input[i]<<std::endl;
  }
}

void Network::load(std::string file) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream.good())
    {
        std::cerr << "Could not load network file, path " << file<< std::endl;
        std::exit(-1);
    }

    const auto num_ft_weights =layers[0].in_features*layers[0].out_features;
    const auto num_ft_bias = layers[0].out_features;
 
    ft_weights = std::make_unique<int16_t[]>(num_ft_weights);
    ft_biases = std::make_unique<int16_t[]>(num_ft_bias);



    int num_weights, num_bias;
    stream.read((char *)&num_weights, sizeof(int));
    weights = std::make_unique<int16_t[]>(num_weights-num_ft_weights);
    stream.read((char *)ft_weights.get(), sizeof(int16_t) * (num_ft_weights));
    stream.read((char *)weights.get(), sizeof(int16_t) * (num_weights-num_ft_weights));
    stream.read((char *)&num_bias, sizeof(int));
    biases = std::make_unique<int32_t[]>(num_bias-num_ft_bias);
    stream.read((char *)ft_biases.get(), sizeof(int16_t) * (num_ft_bias));
    stream.read((char *)biases.get(), sizeof(int32_t) * (num_bias-num_ft_bias));
    stream.close();
  
}

void Network::load_bucket(std::string file){
  //loading the buckets
    std::ifstream stream(file, std::ios::binary);
    if (!stream.good())
    {
        std::cerr << "Could not load network file, path " << file<< std::endl;
        std::exit(-1);
    }
    uint32_t num_hidden,num_buckets;
    stream.read((char*)&num_hidden,sizeof(uint32_t));
    stream.read((char*)&num_buckets,sizeof(uint32_t));
    std::cout<<"NumHidden: "<<num_hidden<<std::endl;
    std::cout<<"NumBuckets: "<<num_buckets<<std::endl;
    std::vector<uint32_t> layer_dims;
    for(auto k=0;k<num_hidden;++k){
      uint32_t hidden;
      stream.read((char*)&hidden,sizeof(uint32_t));
      std::cout<<"hidden: "<<hidden<<std::endl;
      layer_dims.emplace_back(hidden);
    }
    //adding the accumulator
    layers.emplace_back(Layer{120,layer_dims[0]});
    //adding the remaining layers
    for(auto k=0;k<layer_dims.size()-1;++k){
      layers.emplace_back(Layer{layer_dims[k],layer_dims[k+1]});
    }
    layers.emplace_back(Layer{layer_dims[num_hidden-1],1});
    
    //number of weights and biases for the feature transformer
    size_t num_ft_weights,num_hidden_weights;
    size_t num_ft_bias,num_hidden_bias;

    num_ft_weights = layers.front().in_features*layers.front().out_features;
    num_ft_bias = layers.front().out_features;
    std::cout<<"NumFTBias: "<<num_ft_bias<<std::endl;
    //number of weights and biases for the remaining layers
    
    //computing the number of weights
    num_hidden_weights = 0;
    num_hidden_bias = 0;
    for(auto k=1;k<layers.size();++k){
      num_hidden_weights+=layers[k].in_features*layers[k].out_features;
      num_hidden_bias+=layers[k].out_features;
    }
    size_t total_weights = num_ft_weights+num_buckets*num_hidden_weights;
    size_t total_bias = num_ft_bias+num_buckets*num_hidden_bias;
    
    
    weights = std::make_unique<int16_t[]>(total_weights-num_ft_weights);
    biases = std::make_unique<int32_t[]>(total_bias-num_ft_bias);

    ft_weights = std::make_unique<int16_t[]>(num_ft_weights);
    ft_biases = std::make_unique<int16_t[]>(num_ft_bias);

    stream.read((char *)ft_weights.get(), sizeof(int16_t) * (num_ft_weights));
    stream.read((char *)weights.get(), sizeof(int16_t) * (total_weights-num_ft_weights));
    
    stream.read((char *)ft_biases.get(), sizeof(int16_t) * (num_ft_bias));
    stream.read((char *)biases.get(), sizeof(int32_t) * (total_bias-num_ft_bias));
  
    init();

    //initialization goes here

}
void Network::addLayer(Layer layer) {
    layers.emplace_back(layer);
}

void Network::init(){
    if(layers.size()==0){
      return;
    }
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

int32_t Network::get_max_weight() const {
    auto max_value = std::numeric_limits<int16_t>::min();
    size_t num_weights = 0;
    for (Layer l: layers) {
        num_weights += l.out_features * l.in_features;
    }
    for (int i = 0; i < num_weights; ++i) {
        if(std::abs(weights[i])>max_value) {
            max_value = std::abs(weights[i]);
        }
    }

    return max_value;

}

int32_t Network::get_max_bias() const {
    auto max_value = std::numeric_limits<int16_t>::min();
    size_t num_bias= 0;
    for (Layer l: layers) {
        num_bias += l.out_features;
    }
    for (int i = 0; i < num_bias; ++i) {
        if(std::abs(biases[i])>max_value) {
            max_value = std::abs(biases[i]);
        }
    }

    return max_value;

}

int Network::compute_incre_forward_pass(Position next) {
    int16_t *z_previous;

    if (next.color == BLACK) {
        z_previous = accumulator.black_acc.get();
    } else {
        z_previous = accumulator.white_acc.get();
    }
    accumulator.update(next.color,next);
    for (auto i = 0; i < layers[0].out_features; i++) {
        
        auto value = std::clamp(z_previous[i], int16_t{0}, int16_t{127});
        temp[i]=value*value;
        temp[i]=temp[i]/128;
        //temp[i]=value;
    }

    for (auto i = 0; i < layers[0].out_features; ++i)
    {
        input[i] = temp[i];
        temp[i] = 0;
    }
    auto weight_index_offset =0;
    auto bias_index_offset = 0;
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
                auto value= std::clamp(sum/64,  0, 127);
               temp[i]=value*value;
               temp[i]=temp[i]/128;
               //temp[i]=value;
            } else {
                temp[i]= sum/64;
            }
        }


        weight_index_offset += l.out_features * l.in_features;
        bias_index_offset += l.out_features;
        //temp is the input to the next layer
        for(auto p=0; p<l.out_features; ++p) {
            input[p]=temp[p];
        }
    }

    return input[0];
}

int Network::operator[](size_t index) {
    return input[index];
}

int Network::evaluate(Position pos, int ply)
{

    if (pos.BP == 0 && pos.get_color()==BLACK)
    {
        return -loss(ply);
    }
    if (pos.WP == 0 && pos.get_color()==WHITE)
    {
        return loss(ply);
    }

    int32_t val = compute_incre_forward_pass(pos);
    return val;
}

int Network::evaluate(Position pos, int ply, Network &net1, Network &net2) {
    auto num_pieces = Bits::pop_count(pos.BP | pos.WP);
    if (num_pieces>10) {
        return net1.evaluate(pos, ply);
    } else {
        return net2.evaluate(pos, ply);
    }
}





