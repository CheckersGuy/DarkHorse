#include "Bits.h"
#include "incbin.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <emmintrin.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <random>

// finding nonzero values very quickly

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
void find_nnz_version1(int32_t *input, uint16_t *output, uint32_t &count) {
  int nnz = 0;
  for (auto i = 0; i < InDim; ++i) {
    if (input[i] != 0) {
      output[nnz++] = i;
    }
  }
  count = nnz;
}

template <int InDim>
void find_nnz_version2(int32_t *input, uint16_t *output, uint32_t &count) {

  auto *in = reinterpret_cast<const __m256i *>(input);
  constexpr auto num_chnks = InDim / 8;
  int nnz = 0;
  for (auto i = 0; i < num_chnks; ++i) {
    auto c = _mm256_movemask_ps(_mm256_castsi256_ps(
        _mm256_cmpgt_epi32(_mm256_load_si256(in + i), _mm256_setzero_si256())));

    while (c != 0) {
      auto index = Bits::bitscan_foward(c);
      output[nnz++] = index;
      c &= c - 1;
    }
  }
  count = nnz;
}

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

int main() {
  // generate random numbers and measure

  std::mt19937_64 generator;
  std::uniform_real_distribution<double> distrib1(0, 1);
  std::uniform_int_distribution<int> distrib2;

  uint64_t total_time = 0;
  uint64_t total_nnz = 0;
  constexpr auto input_size = 256;
  constexpr auto tries = 100000;
  uint32_t count = 0;
  for (auto k = 0; k < tries; ++k) {
    alignas(64) int32_t input[input_size];
    alignas(64) uint16_t output[input_size];

    for (auto i = 0; i < input_size; ++i) {
      if (distrib1(generator) <= 0.21) {
        input[i] = distrib2(generator);
      } else {
        input[i] = 0;
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    // find_nnz_version1<input_size>(input, output, count);
    find_nnz_version2<input_size>(input, output, count);
    // find_nnz_version3<input_size>(input, output, count);
    auto t2 = std::chrono::high_resolution_clock::now();
    total_nnz += count;
    total_time += (t2 - t1).count();
  }

  std::cout << "totaltime: " << total_time << std::endl;
  std::cout << "totalnnz: " << total_nnz << std::endl;
  auto average = total_time / tries;
  std::cout << "AverageTime: " << average << std::endl;

  return 0;
}
