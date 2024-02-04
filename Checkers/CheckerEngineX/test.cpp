#include "Bits.h"
#include "LinearSparse.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <emmintrin.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <random>
static constexpr int BLOCK_ROWS = 2;
static constexpr int BLOCK_COLS = 2;
static constexpr int INPUT_SIZE = 8; // number of columns in the input matrix
/*
alignas(64) static constexpr std::array<std::array<std::uint16_t, 8>,
                                        256> LookupTableIndices = []() {
  std::array<std::array<std::uint16_t, 8>, 256> v{};
  for (int i = 0; i < 256; ++i) {
    int j = i;
    int k = 0;
    while (j) {
      const auto lsbIndex = Bits::bitscan_foward(std::uint32_t(j));
      j &= j - 1;
      v[i][k] = lsbIndex;
      ++k;
    }
  }
  return v;
}();
*/
int main() {
  // more stuff to learn about. Especially need to have a looko at the
  // lookup-table approach
  constexpr int chunk_size = 4;
  std::mt19937_64 generator(231231ull);
  std::uniform_real_distribution<float> distrib(0, 1);
  std::uniform_int_distribution<uint8_t> distrib2(0, 255);
  // finding the nnz entries efficiently
  const auto input_size = 1024;

  uint16_t num_nnz = 0;
  uint16_t nnz[input_size / chunk_size] = {0};
  alignas(64) uint8_t input[input_size] = {0};

  auto *chunks = reinterpret_cast<const uint32_t *>(input);
  for (auto i = 0; i < input_size; ++i) {
    if (distrib(generator) > 0.93) {
      input[i] = distrib2(generator);
    }
  }
  for (auto i = 0; i < input_size / 4; ++i) {
    std::cout << (int)chunks[i] << ", ";
  }

  std::cout << std::endl;

  auto t1 = std::chrono::high_resolution_clock::now();
  /*for (auto i = 0; i < input_size / chunk_size; ++i) {
    if (chunks[i] != 0) {
      nnz[num_nnz] = i;
      num_nnz++;
    }
  }
  */
  const auto *in = reinterpret_cast<const __m256i *>(input);
  for (auto i = 0; i < input_size / 32; ++i) {
    auto current = _mm256_load_si256(in + i);
    uint32_t nonzeros = _mm256_movemask_ps(
        (__m256)_mm256_cmpgt_epi32(current, _mm256_setzero_si256()));
    while (nonzeros) {
      auto index = Bits::bitscan_foward(nonzeros);
      nonzeros &= nonzeros - 1;
      nnz[num_nnz++] = 8 * i + index;
    }
  }

  auto t2 = std::chrono::high_resolution_clock::now();

  auto dur = (t2 - t1).count();
  std::cout << "TimeTaken: " << dur << std::endl;

  std::cout << "NonZeroEntries" << std::endl;
  for (auto i = 0; i < num_nnz; ++i) {
    std::cout << (int)nnz[i] << ", ";
  }
  return 0;
}
