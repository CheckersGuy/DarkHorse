//
// Created by root on 18.04.21.
//

#include "Network.h"
#include "GameLogic.h"
#include "types.h"
#include <cstdint>
#include <cstdlib>

// Note:: Should look at the accumulator because it's using the input variable
// as well

Layer::Layer(int in, int out, int16_t *w, int32_t *b)
    : in_features(in), out_features(out), weights(w), bias(b) {}

void Accumulator::refresh() {
  for (auto i = 0; i < size; ++i) {
    white_acc[i] = net->ft_biases[i];
    black_acc[i] = net->ft_biases[i];
  }
  previous_black = Position{};
  previous_white = Position{};
}

Accumulator::~Accumulator() {
  free(black_acc);
  free(white_acc);
}

Network::~Network() {
  free(biases);
  free(weights);
  free(ft_biases);
  free(ft_weights);
  free(input);
}

void Accumulator::apply(Color perp, Position before, Position after) {
  int16_t *input = ((perp == BLACK) ? black_acc : white_acc);

  auto WP_O =
      after.get_pieces<WHITE, PAWN>() & (~before.get_pieces<WHITE, PAWN>());
  auto BP_O =
      after.get_pieces<BLACK, PAWN>() & (~before.get_pieces<BLACK, PAWN>());
  auto WK_O =
      after.get_pieces<WHITE, KING>() & (~before.get_pieces<WHITE, KING>());
  auto BK_O =
      after.get_pieces<BLACK, KING>() & (~before.get_pieces<BLACK, KING>());

  size_t offset = 0;

  while (WP_O) {
    auto index = Bits::bitscan_foward(WP_O) - 4 + offset;
    add_feature(input, index);
    WP_O &= WP_O - 1;
  }
  offset += 28;

  while (BP_O) {
    auto index = Bits::bitscan_foward(BP_O) + offset;
    add_feature(input, index);
    BP_O &= BP_O - 1;
  }
  offset += 28;

  while (WK_O) {
    auto index = Bits::bitscan_foward(WK_O) + offset;
    add_feature(input, index);
    WK_O &= WK_O - 1;
  }
  offset += 32;

  while (BK_O) {
    auto index = Bits::bitscan_foward(BK_O) + offset;
    add_feature(input, index);
    BK_O &= BK_O - 1;
  }

  auto WP_Z =
      (~after.get_pieces<WHITE, PAWN>()) & (before.get_pieces<WHITE, PAWN>());
  auto BP_Z =
      (~after.get_pieces<BLACK, PAWN>()) & (before.get_pieces<BLACK, PAWN>());
  auto WK_Z =
      (~after.get_pieces<WHITE, KING>()) & (before.get_pieces<WHITE, KING>());
  auto BK_Z =
      (~after.get_pieces<BLACK, KING>()) & (before.get_pieces<BLACK, KING>());

  offset = 0;

  // to be continued
  while (WP_Z) {
    auto index = Bits::bitscan_foward(WP_Z) - 4 + offset;
    remove_feature(input, index);
    WP_Z &= WP_Z - 1;
  }
  offset += 28;

  while (BP_Z) {
    auto index = Bits::bitscan_foward(BP_Z) + offset;
    remove_feature(input, index);
    BP_Z &= BP_Z - 1;
  }
  offset += 28;

  while (WK_Z) {
    auto index = Bits::bitscan_foward(WK_Z) + offset;
    remove_feature(input, index);
    WK_Z &= WK_Z - 1;
  }
  offset += 32;

  while (BK_Z) {
    auto index = Bits::bitscan_foward(BK_Z) + offset;
    remove_feature(input, index);
    BK_Z &= BK_Z - 1;
  }
}

void Accumulator::update(Color perp, Position after) {
  if (perp == BLACK) {
    apply(perp, previous_black.get_color_flip(), after.get_color_flip());
    previous_black = after;
  } else {
    apply(perp, previous_white, after);
    previous_white = after;
  }
}

void Accumulator::init(Network *net) {
  const auto size = net->layers[0].out_features;
  black_acc =
      (int16_t *)std::aligned_alloc(Network::ALIGNMENT, sizeof(int16_t) * size);
  white_acc =
      (int16_t *)std::aligned_alloc(Network::ALIGNMENT, sizeof(int16_t) * size);

  this->size = size;
  this->net = net;

  for (auto i = 0; i < this->net->layers[0].out_features; ++i) {
    black_acc[i] = net->ft_biases[i];
    white_acc[i] = net->ft_biases[i];
  }
}

void Accumulator::add_feature(int16_t *in, int index) {
  // adding the index-th column to our feature vector
  Simd::vec_add(net->ft_weights + index * net->layers[0].out_features, in,
                size);
}

void Accumulator::remove_feature(int16_t *in, int index) {
  // adding the index-th column to our feature vector
  Simd::vec_diff(net->ft_weights + index * net->layers[0].out_features, in,
                 size);
}

// needs to be rewritten for the new architecture ...
void Network::load_bucket(std::string file) {

  std::ifstream stream(file, std::ios::binary);
  if (!stream.good()) {
    std::cerr << "Could not load network file, path " << file << std::endl;
    std::exit(-1);
  }
  uint32_t num_layers;
  stream.read((char *)&num_layers, sizeof(uint32_t));
  std::vector<std::pair<int, int>> layer_temp;
  for (auto i = 0; i < num_layers; ++i) {
    uint32_t in_features;
    uint32_t out_features;
    stream.read((char *)&in_features, sizeof(uint32_t));
    stream.read((char *)&out_features, sizeof(uint32_t));
    layer_temp.emplace_back(std::make_pair(in_features, out_features));
  }

  // number of weights and biases for the feature transformer
  size_t num_ft_weights, num_hidden_weights;
  size_t num_ft_bias, num_hidden_bias;

  num_ft_weights = layer_temp.front().first * layer_temp.front().second;
  num_ft_bias = layer_temp.front().second;
  // number of weights and biases for the remaining layers

  // computing the number of weights
  num_hidden_weights = 0;
  num_hidden_bias = 0;

  for (auto &[in_features, out_features] : layer_temp) {
    if (in_features == 120) {
      continue;
    }
    num_hidden_weights += in_features * out_features;
    num_hidden_bias += out_features;
  }

  weights = (int16_t *)std::aligned_alloc(
      Network::ALIGNMENT, (num_hidden_weights) * sizeof(int16_t));
  biases = (int32_t *)std::aligned_alloc(Network::ALIGNMENT,
                                         (num_hidden_bias) * sizeof(int32_t));

  ft_weights = (int16_t *)std::aligned_alloc(Network::ALIGNMENT,
                                             num_ft_weights * sizeof(int16_t));
  ft_biases = (int16_t *)std::aligned_alloc(Network::ALIGNMENT,
                                            num_ft_bias * sizeof(int16_t));

  stream.read((char *)ft_weights, sizeof(int16_t) * (num_ft_weights));
  stream.read((char *)weights, sizeof(int16_t) * (num_hidden_weights));

  stream.read((char *)ft_biases, sizeof(int16_t) * (num_ft_bias));
  stream.read((char *)biases, sizeof(int32_t) * (num_hidden_bias));

  int weight_offset = 0;
  int bias_offset = 0;
  int f_in_features = layer_temp.front().first;
  int f_out_features = layer_temp.front().second;
  layers.emplace_back(Layer(f_in_features, f_out_features, nullptr, nullptr));
  for (auto &[in_features, out_features] : layer_temp) {
    if (in_features == 120) {
      continue;
    }
    layers.emplace_back(Layer(in_features, out_features,
                              weights + weight_offset, biases + bias_offset));
    bias_offset += out_features;
    weight_offset += out_features * in_features;
  }

  init();

  // initialization goes here
}
void Network::addLayer(Layer layer) { layers.emplace_back(layer); }

void Network::init() {
  max_units = 0;
  for (Layer l : layers) {
    max_units += l.out_features;
  }
  input = (int16_t *)std::aligned_alloc(Network::ALIGNMENT,
                                        sizeof(int16_t) * max_units);

  af_output = (int32_t *)std::aligned_alloc(Network::ALIGNMENT,
                                            sizeof(int32_t) * max_units);

  accumulator.init(this);
}

int16_t *Network::compute_incre_forward_pass(Position next) {
  int16_t *z_previous;
  if (next.color == BLACK) {
    z_previous = accumulator.black_acc;
  } else {
    z_previous = accumulator.white_acc;
  }
  accumulator.update(next.color, next);
  Simd::accum_activation(z_previous, input, layers[0].out_features);

  // double ratio = ((double)(nnz)) / ((double)layers[0].out_features / 2);
  // std::cout << "Ratio: " << ratio << std::endl;

  int in_offset = 0;
  for (auto k = 1; k < layers.size() - 1; ++k) {
    const Layer &l = layers[k];
    for (auto i = 0; i < l.out_features; i++) {
      int sum = l.bias[i];

      sum += Simd::flatten(l.weights + i * l.in_features, input + in_offset,
                           l.in_features);

      af_output[i + in_offset + l.in_features] = sum / 64;
    }

    Simd::square_clipped(af_output + in_offset + l.in_features,
                         input + in_offset + l.in_features, l.out_features);

    in_offset += l.in_features;
  }
  const Layer &l = layers.back();
  for (auto i = 0; i < l.out_features; i++) {
    int sum = l.bias[i];
    sum += Simd::flatten(l.weights + i * l.in_features, input + in_offset,
                         l.in_features);

    input[i + in_offset + l.in_features] = sum / 64;
  }
  return &input[in_offset + l.in_features];
}

int Network::operator[](int index) { return input[index]; }

int Network::evaluate(Position pos, int ply) {

  if (pos.BP == 0 && pos.get_color() == BLACK) {
    return -loss(ply);
  }
  if (pos.WP == 0 && pos.get_color() == WHITE) {
    return loss(ply);
  }
  return *compute_incre_forward_pass(pos);
}
