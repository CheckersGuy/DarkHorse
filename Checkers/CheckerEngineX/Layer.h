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

  int get_weight_index(int index) {
    // we unroll the outer loop by 4
    const int BLOCK_ROWS = 4;
    const int BLOCK_COLS = 32;

    const int ROW = index / PadInDim;
    const int COL = index % PadInDim;
    const int BLOCK_SIZE = BLOCK_ROWS * BLOCK_COLS;
    int NUM_ROW_BLOCKS = PadInDim / BLOCK_COLS;

    int row = ROW / BLOCK_COLS;
    int col = COL / BLOCK_COLS;
    //  std::cout << "RowBlock: " << row << std::endl;
    //  std::cout << "RowCol:" << col << std::endl;
    int block_row = row % BLOCK_ROWS;
    int block_col = row % BLOCK_COLS;

    // std::cout << "BlockRow: " << block_row << std::endl;
    // std::cout << "BlockCol: " << block_col << std::endl;
    // std::cout << "Index: (" << row << ", " << col << ")" << std::endl;

    return (PadInDim / BLOCK_COLS) * row * BLOCK_SIZE + col * BLOCK_SIZE +
           block_row * BLOCK_ROWS + block_col;
  }

  void load_params(std::ifstream &stream) {
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

          // temp_weights[i * PadInDim + j] = weight;

          weights[i * PadInDim + j] = weight;
        }
      }
      // reordering weights

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
