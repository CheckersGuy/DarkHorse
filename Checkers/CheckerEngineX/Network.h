//
// Created by root on 18.04.21.
//

#ifndef READING_NETWORK_H
#define READING_NETWORK_H

#include "Position.h"
#include "Simd.h"
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
  const int16_t *weights;
  const int32_t *bias;

  Layer(int in, int out, int16_t *weights, int32_t *bias);
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

struct Network {
  constexpr static size_t ALIGNMENT = 32;
  std::vector<Layer> layers;
  int16_t *ft_biases;
  int16_t *ft_weights;
  int32_t *biases;
  int16_t *weights;
  int16_t *input;     // input or output of activations
  int32_t *af_output; // affine transform output
  int max_units{0};
  Accumulator accumulator;

  ~Network();

  void addLayer(Layer layer);

  void load_bucket(std::string file);

  void init();

  int16_t *compute_incre_forward_pass(Position next);

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
