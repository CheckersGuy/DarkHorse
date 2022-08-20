//
// Created by root on 18.04.21.
//

#include <complex>
#include "Network.h"

void Accumulator::update(Color perp, Position before,Position after) {

    if(perp == BLACK){
        before = before.get_color_flip();
        after = after.get_color_flip();
    }

    //computing the changes

    add_feature(perp,before,after);
    remove_feature(perp,before,after);
    

}

void Accumulator::add_feature(Color perp,size_t index){
    //adding the index-th column to our feature vector
    float * input = ((perp == BLACK)? black_acc.get() : white_acc.get());
    for(auto i=0;i<network.layers[0].out_features;++i){
        input[i] += network.weights[index*network.layers[0].out_features+i];
    }
}

void Accumulator::remove_feature(Color perp,size_t index){
    //adding the index-th column to our feature vector
    float * input = ((perp == BLACK)? black_acc.get() : white_acc.get());
    for(auto i=0;i<network.layers[0].out_features;++i){
        input[i] -= network.weights[index*network.layers[0].out_features+i];
    }
}

void Accumulator::add_feature(Color perp, Position& before, Position&after){
    //need to check this
    auto WP = after.WP&(~before.WP);
    auto BP = after.BP&(~before.BP);
    auto WK = (after.WP & after.K)&(~(before.WP & before.K));
    auto BK = (after.BP & after.K)&(~(before.BP & before.K));

    //to be continued
    while (WP){
        auto index = Bits::bitscan_foward(WP) - 4;
        add_feature(perp,index);
        WP &= WP - 1;
    }

     while (BP){
        auto index = Bits::bitscan_foward(WP) ;
        BP &= BP - 1;
    }

     while (WK){
        auto index = Bits::bitscan_foward(WK);
        WK &= WK - 1;
    }

     while (BK){
        auto index = Bits::bitscan_foward(BK);
        BK &= BK - 1;
    }
}

void Accumulator::remove_feature(Color perp, Position &before, Position &after){
    auto WP = before.WP&(~after.WP);
    auto BP = before.BP&(~after.BP);
    auto WK = (before.WP & before.K)&(~(after.WP & after.K));
    auto BK = (before.BP & before.K)&(~(after.BP & after.K));
}



Accumulator &Accumulator::operator=(const Accumulator &other)
{
    size = other.size;
    black_acc = std::make_unique<float[]>(size);
    white_acc = std::make_unique<float[]>(size);

    std::copy(other.black_acc.get(), other.black_acc.get() + size, black_acc.get());
    std::copy(other.white_acc.get(), other.white_acc.get() + size, white_acc.get());
    return *this;
}

std::pair<uint32_t, uint32_t> compute_difference(uint32_t previous, uint32_t next) {
    //computes the indices which need to be either set to 0
    //or for which we need to compute the matrix-vec multiply
    uint32_t changed_to_ones = next & (~previous);
    uint32_t n_next = ~next;
    uint32_t changed_to_zeros = n_next & (previous);
    return std::make_pair(changed_to_zeros, changed_to_ones);
}

void Network::load(std::string file) {
    std::ifstream stream(file, std::ios::binary);
    if (!stream.good()) {
        std::cerr << "Could not load the weights" << std::endl;
        std::exit(-1);
    }


    int num_weights, num_bias;
    stream.read((char *) &num_weights, sizeof(int));
    weights = std::unique_ptr<float[]>((float *) _mm_malloc(sizeof(float) * num_weights, 32));;
    stream.read((char *) weights.get(), sizeof(float) * num_weights);
    stream.read((char *) &num_bias, sizeof(int));
    biases = std::unique_ptr<float[]>((float *) _mm_malloc(sizeof(float) * num_bias, 32));
    stream.read((char *) biases.get(), sizeof(float) * num_bias);
    stream.close();

}

void Network::addLayer(Layer layer) {
    layers.emplace_back(layer);
}

void Network::init() {

    p_black = Position{};
    p_white = Position{};

    for (Layer l: layers)
        max_units = std::max(std::max(l.in_features, l.out_features), max_units);

    temp = std::unique_ptr<float[]>((float *) _mm_malloc(sizeof(float) * max_units, 32));
    input = std::unique_ptr<float[]>((float *) _mm_malloc(sizeof(float) * max_units, 32));

    for (auto i = 0; i < max_units; ++i) {
        temp[i] = 0;
        input[i] = 0;
    }
    const size_t units = layers[0].out_features;
    z_black = std::unique_ptr<float[]>((float *) _mm_malloc(sizeof(float) * units, 32));
    z_white = std::unique_ptr<float[]>((float *) _mm_malloc(sizeof(float) * units, 32));
    //initializing those two vectors
    for (auto i = 0; i < units; ++i) {
        z_black[i] = biases[i];
        z_white[i] = biases[i];
    }

}

float Network::get_max_weight() {
    float max_value = std::numeric_limits<float>::min();
    size_t num_weights = 0;
    for (Layer l: layers) {
        num_weights += l.out_features * l.in_features;
    }
    for (int i = 0; i < num_weights; ++i) {
        max_value = std::max(std::abs(weights[i]), max_value);
    }

    return max_value;

}

float Network::compute_incre_forward_pass(Position next) {
    //to be continued
    Color color = next.color;
    float *z_previous;
    Position previous;
    if (color == BLACK) {
        previous = p_black;
        z_previous = z_black.get();
        next = next.get_color_flip();
        p_black = next;
    } else {
        previous = p_white;
        z_previous = z_white.get();
        p_white = next;
    }

    size_t weight_index_offset = 0u;
    size_t bias_index_offset = 0u;
    size_t offset = 0u;

    auto diff_men_b = compute_difference(previous.BP & (~previous.K), next.BP & (~next.K));
    auto diff_men_w = compute_difference(previous.WP & (~previous.K), next.WP & (~next.K));
    auto diff_men_bk = compute_difference(previous.BP & (previous.K), next.BP & (next.K));
    auto diff_men_wk = compute_difference(previous.WP & (previous.K), next.WP & (next.K));
    {
        uint32_t n = diff_men_w.second;
        uint32_t p = diff_men_w.first;


        while (n != 0) {
            auto j = Bits::bitscan_foward(n) + offset - 4;
            for (auto i = 0; i < layers[0].out_features; ++i) {
                z_previous[i] += weights[weight_index_offset + j * layers[0].out_features + i];
            }
            n &= n - 1u;
        }
        while (p != 0) {
            auto j = Bits::bitscan_foward(p) + offset - 4;
            for (auto i = 0; i < layers[0].out_features; ++i) {
                z_previous[i] -= weights[weight_index_offset + j * layers[0].out_features + i];
            }
            p &= p - 1u;
        }
        offset += 28;
    }
    {
        uint32_t n = diff_men_b.second;
        uint32_t p = diff_men_b.first;

        while (n != 0) {
            auto j = Bits::bitscan_foward(n) + offset;
            for (auto i = 0; i < layers[0].out_features; ++i) {
                z_previous[i] += weights[weight_index_offset + j * layers[0].out_features + i];
            }
            n &= n - 1u;
        }

        while (p != 0) {
            auto j = Bits::bitscan_foward(p) + offset;
            for (auto i = 0; i < layers[0].out_features; ++i) {
                z_previous[i] -= weights[weight_index_offset + j * layers[0].out_features + i];
            }
            p &= p - 1u;
        }
        offset += 28;
    }

    {
        uint32_t n = diff_men_wk.second;
        uint32_t p = diff_men_wk.first;

        while (n != 0) {
            auto j = Bits::bitscan_foward(n) + offset;
            for (auto i = 0; i < layers[0].out_features; ++i) {
                z_previous[i] += weights[weight_index_offset + j * layers[0].out_features + i];
            }
            n &= n - 1u;
        }

        while (p != 0) {
            auto j = Bits::bitscan_foward(p) + offset;
            for (auto i = 0; i < layers[0].out_features; ++i) {
                z_previous[i] -= weights[weight_index_offset + j * layers[0].out_features + i];
            }
            p &= p - 1u;
        }
        offset += 32;
    }

    {
        uint32_t n = diff_men_bk.second;
        uint32_t p = diff_men_bk.first;

        while (n != 0) {
            auto j = Bits::bitscan_foward(n) + offset;
            for (auto i = 0; i < layers[0].out_features; ++i) {
                z_previous[i] += weights[weight_index_offset + j * layers[0].out_features + i];
            }
            n &= n - 1u;
        }

        while (p != 0) {
            auto j = Bits::bitscan_foward(p) + offset;
            for (auto i = 0; i < layers[0].out_features; ++i) {
                z_previous[i] -= weights[weight_index_offset + j * layers[0].out_features + i];
            }
            p &= p - 1u;
        }
    }
    for (auto i = 0; i < layers[0].out_features; i++) {
        temp[i] = std::clamp(z_previous[i], 0.0f, 1.0f);
    }
    memcpy(input.get(), temp.get(), sizeof(float) * layers[0].out_features);
    memset(temp.get(), 0, sizeof(float) * layers[0].out_features);

    weight_index_offset += layers[0].out_features * layers[0].in_features;
    bias_index_offset += layers[0].out_features;
    //computation for the remaining layers
    for (auto k = 1; k < layers.size(); ++k) {
        Layer l = layers[k];
        for (auto j = 0; j < l.in_features; ++j) {
            for (auto i = 0; i < l.out_features; i++) {
                temp[i] += weights[weight_index_offset + j * l.out_features + i] * input[j];
            }
        }

        for (auto i = 0; i < l.out_features; ++i) {
            temp[i] += biases[bias_index_offset + i];
            if (k < layers.size() - 1) {
                temp[i] = std::clamp(temp[i], 0.0f, 1.0f);
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

float *Network::get_output() {
    return input.get();
}

float Network::forward_pass() const {

    size_t weight_index_offset = 0u;
    size_t bias_index_offset = 0u;
    for (auto k = 0; k < layers.size(); ++k) {
        const Layer &l = layers[k];
        for (auto j = 0; j < l.in_features; ++j) {
            if (input[j] == 0)
                continue;
            for (auto i = 0; i < l.out_features; ++i) {
                temp[i] += weights[weight_index_offset + j * l.out_features + i] * input[j];
            }
        }
        for (auto i = 0; i < l.out_features; ++i) {
            temp[i] += biases[bias_index_offset + i];
            if (k < layers.size() - 1) {
                temp[i] = std::clamp(temp[i], 0.0f, 1.0f);
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

int Network::evaluate(Position pos, int ply) {

    if (pos.BP == 0) {
        return -pos.color * loss(ply);
    }
    if (pos.WP == 0) {
        return pos.color * loss(ply);
    }


    float val = compute_incre_forward_pass(pos) * 64.0f;
    return static_cast<int>(val);
}

int Network::evaluate(Position pos, int ply, Network &net1, Network &net2) {
    auto num_pieces = Bits::pop_count(pos.BP | pos.WP);
    if (num_pieces <= 24 && num_pieces>=12) {
        return net1.evaluate(pos, ply);
    } else {
        return net2.evaluate(pos, ply);
    }
}

void Network::set_input(Position p) {
    //testing another network architecture
    if (p.color == BLACK) {
        p = p.get_color_flip();
    }
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
}

float Network::get_win_p(Position pos){
    auto value = compute_incre_forward_pass(pos);
    return 1.0/(1.0+std::exp(-value));
}

