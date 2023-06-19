//
// Created by root on 18.04.21.
//

#ifndef READING_NETWORK_H
#define READING_NETWORK_H

#include "Layer.h"
#include "Position.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <ranges>
#include <vector>

class Network;

struct Layer {
  const int in_features;
  const int out_features;
  const int8_t *weights;
  const int32_t *bias;

  Layer(int in, int out, int8_t *weights, int32_t *bias);
};

struct Accumulator {
  static constexpr int OutDim = 1512;
  alignas(64) int16_t black_acc[OutDim] = {0};
  alignas(64) int16_t white_acc[OutDim] = {0};
  int16_t *ft_biases;
  int16_t *ft_weights;

  int size;
  Position previous_black, previous_white;

  ~Accumulator();

  void update(Color per, Position after);

  inline void add_feature(int16_t *in, int index) {
    Simd::vec_add<OutDim>(ft_weights + index * OutDim, in);
  }

  inline void remove_feature(int16_t *in, int index) {
    Simd::vec_diff<OutDim>(ft_weights + index * OutDim, in);
  }

  void apply(Color color, Position before, Position after);

  void refresh();

  void load_weights(std::ifstream &stream);
};

struct Network {
  constexpr static size_t ALIGNMENT = 64;
  std::vector<Layer> layers;
  int max_units{0};
  Accumulator accumulator;
  QLayer<1512, 16, Activation::SqRelu> first;
  QLayer<16, 32, Activation ::SqRelu> second;
  QLayer<32, 1> output;
  alignas(64) int8_t input[2048 + 32 + 32 + 1] = {0};

  void load_bucket(std::string file);

  int32_t *compute_incre_forward_pass(Position next);

  int evaluate(Position pos, int ply);

  int operator[](int index);

  friend class Accumulator;

  friend std::ostream &operator<<(std::ostream &stream, const Network &other) {
    stream << "Num_Layers: " << other.layers.size() << std::endl;
    for (auto &layer : other.layers) {
      stream << "Layer: "
             << "InFeatures: " << layer.in_features
             << " OutFeatures: " << layer.out_features << std::endl;
      ;
    }

    return stream;
  }
};

#endif // READING_NETWORK_H
