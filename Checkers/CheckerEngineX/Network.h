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

struct Accumulator {
  static constexpr int OutDim = 2 * 1024;
  alignas(64) int16_t black_acc[OutDim] = {0};
  alignas(64) int16_t white_acc[OutDim] = {0};
  int16_t *ft_biases;
  int16_t *ft_weights;

  int size;
  Position previous_black, previous_white;
  std::array<int, 32> active_features;
  std::array<int, 32> removed_features;

  ~Accumulator();

  void update(Color per, Position after);

  inline void add_feature(int16_t *in, int index) {
    // adding the index-th column to our feature vector
    Simd::vec_add<OutDim>(ft_weights + index * OutDim, in);
  }

  inline void remove_feature(int16_t *in, int index) {
    // adding the index-th column to our feature vector
    Simd::vec_diff<OutDim>(ft_weights + index * OutDim, in);
  }

  void apply(Color color, Position before, Position after);

  void refresh();

  void load_weights(std::ifstream &stream);

  uint8_t *forward(uint8_t *in, const Position &next);
};

struct Network {
  constexpr static size_t ALIGNMENT = 64;
  int max_units{0};
  Accumulator accumulator;
  QLayer<1024, 16, Activation::SqRelu> first;
  QLayer<16, 32, Activation ::SqRelu> second;
  QLayer<32, 1> output;
  alignas(64) uint8_t input[2048 + 32 + 32 + 1] = {0};

  void load_bucket(std::string file);

  int32_t *compute_incre_forward_pass(Position next);

  int evaluate(Position pos, int ply);

  int operator[](int index);

  friend class Accumulator;
};

#endif // READING_NETWORK_H
