
// #include "Network.h"

#ifndef LinearSp
#define LinearSp
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

template <int InDim>
void find_nnz_version1(int32_t *input, uint16_t *output, uint32_t &count) {
  int nnz = 0;
  for (auto i = 0; i < InDim; ++i) {
    if (input[i] != 0) {
      output[nnz++] = i;
    }
  }
  count = nnz;
}

#define AVX256

template <int InDim, int OutDim> struct SparseLayer {
  static_assert(OutDim % 16 == 0);

  constexpr static int ceil_to_multi(int numToRound, int multiple) {
    if (multiple == 0)
      return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
      return numToRound;

    return numToRound + multiple - remainder;
  }

#ifdef AVX256
  static constexpr int PadInDim = ceil_to_multi(InDim, 32);
  static constexpr int PadOutDim = ceil_to_multi(OutDim, 32);
  // placeholder
  static constexpr int CACHE_LINE_SIZE = 64;
  static constexpr int CHUNKSIZE = 4;
#endif

  alignas(CACHE_LINE_SIZE) int32_t biases[OutDim * NUM_BUCKETS];
  alignas(CACHE_LINE_SIZE) int8_t weights[PadInDim * OutDim * NUM_BUCKETS];
  alignas(CACHE_LINE_SIZE) int32_t buffer[PadOutDim] = {0};

  alignas(CACHE_LINE_SIZE) uint16_t nnz[InDim / CHUNKSIZE];

  int get_weight_index(int index) {
    const int BLOCK_ROWS = OutDim;
    const int BLOCK_COLS = CHUNKSIZE;

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

  static constexpr int get_weight_index_scrambled(int i) {
    return (i / CHUNKSIZE) % (PadInDim / CHUNKSIZE) * OutDim * CHUNKSIZE +
           i / PadInDim * CHUNKSIZE + i % CHUNKSIZE;
  }

  void load_params(std::istream &stream) {
    for (auto k = 0; k < NUM_BUCKETS; ++k) {
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
    }
  }

  auto *forward(uint8_t *input, int bucket_index) {
    const auto w_offset = bucket_index * PadInDim * OutDim;
    const auto b_offset = bucket_index * OutDim;
    // number of output registers for avx2

    const auto numRegs = OutDim / 8;
    __m256i out_regs[numRegs];
    int32_t *input32 = reinterpret_cast<int32_t *>(input);

    // computing nonzero-input-indices

    uint32_t count = 0;
    find_nnz_version3<InDim / CHUNKSIZE>(input32, nnz, count);

    // loading the biases
    const __m256i *biasvec =
        reinterpret_cast<const __m256i *>(biases + b_offset);
    for (auto i = 0; i < numRegs; ++i) {
      out_regs[i] = biasvec[i];
    }

    for (auto j = 0; j < count; ++j) {
      const auto i = nnz[j];
      const auto in = _mm256_set1_epi32(input32[i]);
      const __m256i *col = reinterpret_cast<const __m256i *>(
          &weights[i * OutDim * CHUNKSIZE + w_offset]);
      for (auto k = 0; k < numRegs; ++k) {
        Simd::m256_add_dpbusd_epi32(out_regs[k], in, col[k]);
      }
    }
    // copying to output buffer
    __m256i *outptr = reinterpret_cast<__m256i *>(buffer);
    for (auto k = 0; k < numRegs; ++k) {
      outptr[k] = _mm256_srai_epi32(out_regs[k], 6);
    }

    // computing the activation
    auto *out = input + PadInDim;
    Simd::clipped8<PadOutDim>(&buffer[0], out);
    return out;
    // activation and other things
  }
};

#endif // !LinearSp
