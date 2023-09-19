// #include "Network.h"
#include "Simd.h"
#include "types.h"
#include <algorithm>
#include <cstdint>
#include <emmintrin.h>
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

  alignas(CACHE_LINE_SIZE) int32_t biases[OutDim * NUM_BUCKETS];
  alignas(CACHE_LINE_SIZE) int8_t weights[PadInDim * OutDim * NUM_BUCKETS];
  alignas(CACHE_LINE_SIZE) int32_t buffer[PadOutDim] = {0};

  int get_weight_index(int index) {

    const int BLOCK_ROWS = 4;
    const int BLOCK_COLS = 32;

    const int ROW = index / PadInDim;
    const int COL = index % PadInDim;
    const int BLOCK_SIZE = BLOCK_ROWS * BLOCK_COLS;
    int NUM_ROW_BLOCKS = PadInDim / BLOCK_COLS;

    int row = ROW / BLOCK_ROWS;
    int col = COL / BLOCK_COLS;

    int block_row = ROW % BLOCK_ROWS;
    int block_col = COL % BLOCK_COLS;

    int out = (PadInDim / BLOCK_COLS) * row * BLOCK_SIZE + col * BLOCK_SIZE +
              block_row * BLOCK_COLS + block_col;
    return out;
  }

  void load_params(std::ifstream &stream) {
    for (auto k = 0; k < NUM_BUCKETS; ++k) {
      if constexpr ((OutDim % 4) == 0) {
        int8_t temp_weights[PadInDim * OutDim] = {0};
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
        for (int i = 0; i < PadInDim * OutDim; ++i) {
          auto index = get_weight_index(i);
          weights[index + k * PadInDim * OutDim] = temp_weights[i];
        }
        stream.read((char *)&biases[0 + k * OutDim], sizeof(int32_t) * OutDim);
      } else {
        for (auto i = 0; i < OutDim; ++i) {
          for (auto j = 0; j < PadInDim; ++j) {
            int8_t weight;
            if (j < InDim) {
              stream.read((char *)&weight, sizeof(int8_t));
            } else {
              weight = 0;
            }

            weights[i * PadInDim + j + k * PadInDim * OutDim] = weight;
          }
        }
        stream.read((char *)&biases[0 + k * OutDim], sizeof(int32_t) * OutDim);
      }
    }
  }
  auto *forward(uint8_t *input, int bucket_index) {
    const auto w_offset = bucket_index * PadInDim * OutDim;
    if constexpr ((OutDim % 4) != 0) {
      for (auto i = 0; i < OutDim; ++i) {
        int sum = biases[i + bucket_index * OutDim];
        sum +=
            Simd::flatten8<PadInDim>(weights + PadInDim * i + w_offset, input);
        buffer[i] = sum / 64;
      }
    } else {
      constexpr int in_chunks = PadInDim / 32; // number of blocks in a  row
      constexpr int out_chunks = OutDim / 4;   // number of blocks in a column
      const auto *in = reinterpret_cast<const __m256i *>(input);
      const auto *bias =
          reinterpret_cast<const __m128i *>(biases + bucket_index * OutDim);
      auto *out = reinterpret_cast<__m128i *>(buffer);

      for (auto i = 0; i < out_chunks; ++i) {
        __m256i acc[4] = {_mm256_setzero_si256()};
        int block_index = i * PadInDim * 4;
        for (auto j = 0; j < in_chunks; ++j) {
          const auto in_reg = _mm256_load_si256(in + j);
          const auto weight_index = block_index + j * 128;

          const auto *weight_vec = reinterpret_cast<const __m256i *>(
              weights + weight_index + w_offset);
          for (auto k = 0; k < 4; ++k) {
            const auto weight = _mm256_load_si256(&weight_vec[k]);
            Simd::m256_add_dpbusd_epi32(acc[k], in_reg, weight);
          }
        }
        const __m128i b = _mm_load_si128(bias + i);
        auto out_val = Simd::m256_haddx4(acc[0], acc[1], acc[2], acc[3], b);
        out_val = _mm_srai_epi32(out_val, 6);
        _mm_store_si128(out + i, out_val);
      }
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
