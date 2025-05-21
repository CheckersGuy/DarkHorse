#include "Bits.h"
#include "Simd.h"
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
int main() {
  // generate random numbers and measure

  std::mt19937_64 generator(3231423131ull);
  std::uniform_int_distribution<int16_t> distrib(-5000, 5000);

  constexpr int input_size = 0;
  constexpr int CACHE_LINE_SIZE = 64;

  // getting some

  alignas(CACHE_LINE_SIZE) int16_t input[input_size];
  alignas(CACHE_LINE_SIZE) uint8_t output[input_size];

  for (auto i = 0; i < input_size; ++i) {
    input[i] = distrib(generator);
  }

  Simd::accum_activation8<input_size>(input, output);

  return 0;
}
