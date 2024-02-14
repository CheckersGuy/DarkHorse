
// #include "Network.h"
#include "Bits.h"
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

static constexpr int lsb_constexpr(std::uint32_t v) {
  int c = 0;
  if (!v)
    return 32;
  while (!(v & 1)) {
    v >>= 1;
    ++c;
  }
  return c;
}

alignas(64) static inline const
    std::array<std::array<std::uint16_t, 8>, 256> lookup_indices = []() {
      std::array<std::array<std::uint16_t, 8>, 256> v{};
      for (unsigned i = 0; i < 256; ++i) {
        std::uint64_t j = i, k = 0;
        while (j) {
          v[i][k++] = lsb_constexpr(j);
          j &= j - 1;
        }
      }
      return v;
    }();

template <int InDim>
void find_nnz_version3(int32_t *input, uint16_t *output, uint32_t &count_out) {

  auto *in = reinterpret_cast<const __m256i *>(input);
  constexpr auto num_chnks = InDim / 8;
  uint32_t count = 0;
  auto base = _mm_setzero_si128();
  const auto increment = _mm_set1_epi16(8);
  for (auto i = 0; i < num_chnks; ++i) {
    auto nnz = _mm256_movemask_ps(_mm256_castsi256_ps(
        _mm256_cmpgt_epi32(_mm256_load_si256(in + i), _mm256_setzero_si256())));

    const auto offsets =
        _mm_load_si128(reinterpret_cast<const __m128i *>(&lookup_indices[nnz]));
    _mm_storeu_si128(reinterpret_cast<__m128i *>(output + count),
                     _mm_add_epi16(base, offsets));
    count += Bits::pop_count(nnz);
    base = _mm_add_epi16(base, increment);
  }
  count_out = count;
}

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
  static_assert(OutDim % 4 == 0);

#ifdef AVX256
  static constexpr int PadInDim = ceil_to_multi(InDim, 32);
  static constexpr int PadOutDim = ceil_to_multi(OutDim, 32);
  // placeholder
  static constexpr int CACHE_LINE_SIZE = 64;
  static constexpr int CHUNKSIZE = 4;
#endif

  alignas(CACHE_LINE_SIZE) int32_t biases[OutDim];
  alignas(CACHE_LINE_SIZE) int8_t weights[PadInDim * OutDim];
  alignas(CACHE_LINE_SIZE) int32_t buffer[PadOutDim] = {0};
  alignas(CACHE_LINE_SIZE) uint16_t nnz[ceil_to_multi(OutDim / CHUNKSIZE, 16)];

  int get_weight_index(int index) {

    const int BLOCK_ROWS = OutDim;
    const int BLOCK_COLS = 4;

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

  void load_params(std::istream &stream) {
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
    const auto *in = reinterpret_cast<const __m256i *>(input);
    const auto *bias = reinterpret_cast<const __m256i *>(biases);
    auto *out = reinterpret_cast<__m256i *>(buffer);

    // number of output registers for avx2

    const auto numRegs = OutDim / 32;
    __m256i out_regs[numRegs];
    int32_t *input32 = reinterpret_cast<int32_t *>(input);

    // computing nonzero-input-indices

    uint32_t count = 0;
    find_nnz_version3<OutDim / CHUNKSIZE>(input32, nnz, count);

    // loading the biases
    for (auto i = 0; i < numRegs; ++i) {
      out_regs[i] = bias[i];
    }

    for (auto i = 0; i < count; ++i) {
      auto *col =
          reinterpret_cast<const __m256i *>(&weights[i * OutDim * CHUNKSIZE]);
      const auto in = _mm256_set1_epi32(input32[nnz[i]]);
      for (auto k = 0; k < numRegs; ++k) {
        Simd::m256_add_dpbusd_epi32(out_regs[k], in, col[k]);
      }
    }
    // copying to output buffer

    for (auto k = 0; k < numRegs; ++k) {
      out[k] = out_regs[k];
    }

    if constexpr (ac == Id) {
      return &buffer[0];
    } else {
      // computing the activation
      auto *out = input + PadInDim;
      Simd::clipped8<PadOutDim>(&buffer[0], out);
      return out;
    }
    // activation and other things
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
