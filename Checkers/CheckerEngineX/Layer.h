// #include "Network.h"
#include "Simd.h"
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <immintrin.h>
#include <iostream>
// for testing purposes

enum Activation { Id, SqRelu };

#define AVX256

constexpr int ceil_to_multi(int numToRound, int multiple) {
  if (multiple == 0)
    return numToRound;

  int remainder = numToRound % multiple;
  if (remainder == 0)
    return numToRound;

  return numToRound + multiple - remainder;
}

template <int InDim, int OutDim, Activation ac = Id> struct QLayer {

#ifdef AVX256
  static constexpr int PadInDim = ceil_to_multi(InDim, 32);
  static constexpr int PadOutDim = ceil_to_multi(OutDim, 32);
  // placeholder
  static constexpr int CACHE_LINE_SIZE = 64;
#endif

  alignas(CACHE_LINE_SIZE) int32_t biases[OutDim];
  alignas(CACHE_LINE_SIZE) int8_t weights[PadInDim * OutDim];
  alignas(CACHE_LINE_SIZE) int32_t buffer[OutDim] = {0};

  void load_params(std::ifstream &stream) {
    // loading biases and weights
    // loading the weights not sure if correct

    // specialization if the output-dimensions is divisible by 4
    // then we do blocked-mat-mul

    if constexpr ((OutDim % 4) == 0) {
      int8_t temp_weights[PadInDim * PadOutDim];
      for (auto i = 0; i < OutDim; ++i) {
        for (auto j = 0; j < PadInDim; ++j) {
          int8_t weight;
          if (j < InDim) {
            stream.read((char *)&weight, sizeof(int8_t));
          } else {
            weight = 0;
          }

          temp_weights[i * PadInDim + j] = weight;
        }
      }
      // reordering the weights
      // to be continued

    } else {

      for (auto i = 0; i < OutDim; ++i) {
        for (auto j = 0; j < PadInDim; ++j) {
          int8_t weight;
          if (j < InDim) {
            stream.read((char *)&weight, sizeof(int8_t));
          } else {
            weight = 0;
          }

          weights[i * PadInDim + j] = weight;
        }
      }
    }

    stream.read((char *)&biases[0], sizeof(int32_t) * OutDim);
  }
  auto *forward(int8_t *input) {
    // implement loading the network first
    // then the base case
    // then testing
    // then AVX2
    for (auto i = 0; i < OutDim; ++i) {
      int sum = biases[i];
      sum += Simd::flatten8<PadInDim>(weights + PadInDim * i, input);
      buffer[i] = sum / 64;
    }

    if constexpr (ac == Id) {
      return &buffer[0];
    } else {
      // computing the activation
      auto *out = input + PadInDim;
      Simd::square_clipped8<PadOutDim>(&buffer[0], out);
      return out;
    }
  }

  friend std::ostream &operator<<(std::ostream &stream, const QLayer &layer) {
    std::cout << "PadInDim: " << PadInDim << std::endl;
    std::cout << "InDim: " << InDim << std::endl;
    std::cout << "OutDim: " << OutDim << std::endl;
    for (auto i = 0; i < OutDim; ++i) {
      for (auto j = 0; j < PadInDim; ++j) {
        auto weight = layer.weights[i * PadInDim + j];
        std::cout << (int)weight << ", ";
      }
      std::cout << "\n";
    }
    std::cout << "BIAS" << std::endl;
    for (auto i = 0; i < OutDim; ++i) {
      std::cout << layer.biases[i] << std::endl;
    }
    return stream;
  }
};
