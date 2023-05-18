//
// Created by root on 18.04.21.
//

#ifndef READING_NETWORK_H
#define READING_NETWORK_H

#include "Position.h"
#include "Simd.h"
#include <cmath>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <vector>
class Network;

struct Layer {
  const int in_features;
  const int out_features;

  Layer(int in, int out);
};

struct Accumulator {
  int16_t *black_acc;
  int16_t *white_acc;
  size_t size;
  Position previous_black, previous_white;
  Network *net = nullptr;

  ~Accumulator();

  void init(Network *net);

  void update(Color per, Position after);

  void add_feature(int16_t *input, int index);

  void remove_feature(int16_t *input, int index);

  void apply(Color color, Position before, Position after);

  void refresh();
};

template <typename T> void display_network_data(std::string network_file) {
  std::ifstream stream(network_file, std::ios::binary);
  if (!stream.good()) {
    std::cerr << "Could not load the weights" << std::endl;
    std::exit(-1);
  }

  int num_weights, num_bias;
  stream.read((char *)&num_weights, sizeof(int));
  auto weights = std::make_unique<T[]>(num_weights);

  stream.read((char *)weights.get(), sizeof(T) * num_weights);
  stream.read((char *)&num_bias, sizeof(int));
  auto biases = std::make_unique<T[]>(num_bias);
  stream.read((char *)biases.get(), sizeof(T) * num_bias);

  for (auto i = 0; i < num_bias; ++i) {
    std::cout << biases[i] << std::endl;
  }

  stream.close();
}

struct Network {
  constexpr static size_t ALIGNMENT = 32;
  std::vector<Layer> layers;
  int16_t *ft_biases;
  int16_t *ft_weights;
  int32_t *biases;
  int16_t *weights;
  int16_t *input;
  int bucket_bias_offset = 0;
  int bucket_weight_offset = 0;
  int max_units{0};
  size_t num_buckets{0};
  Accumulator accumulator;

  ~Network();

  int32_t get_max_weight() const;

  int32_t get_max_bias() const;

  void addLayer(Layer layer);

  void load_bucket(std::string file);

  void init();

  void print_output_layer();

  int compute_incre_forward_pass(Position next, int bucket_index);

  int compute_incre_forward_pass(Position next);

  int evaluate(Position pos, int ply);

  int operator[](int index);

  friend class Accumulator;

  friend std::ostream &operator<<(std::ostream &stream, const Network &other) {
    stream << "Num_Layers: " << other.layers.size() << std::endl;
    stream << "Num_Buckets: " << other.num_buckets << std::endl;
    for (auto &layer : other.layers) {
      stream << "Layer: "
             << "InFeatures: " << layer.in_features
             << " OutFeatures: " << layer.out_features << std::endl;
      ;
    }

    return stream;
  }

  void print_bucket_evals(Position next);
};

void testing_simd_functions();
#endif // READING_NETWORK_H
