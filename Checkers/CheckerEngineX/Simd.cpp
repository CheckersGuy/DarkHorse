// moresimd stuff
#include <bits/chrono.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <new>
#define AVX

inline void accum_activation(int16_t *input, const int input_size) {
#ifdef AVX

  const int half_input = input_size / 2;
  const int num_chunks = input_size / 16;
  const int half_chunks = num_chunks / 2;

  auto *in_a = reinterpret_cast<const __m256i *>(input);
  auto *output = reinterpret_cast<__m256i *>(input);
  auto *in_b = reinterpret_cast<const __m256i *>(input + half_input);
  const auto max_val = _mm256_set1_epi16(127);
  const auto min_val = _mm256_set1_epi16(0);
  for (auto i = 0; i < num_chunks; ++i) {
    auto t_a = _mm256_load_si256(in_a + i);
    auto out = _mm256_slli_epi16(t_a, 2);
    out = _mm256_max_epi16(out, min_val);
    out = _mm256_min_epi16(out, max_val);
    out = _mm256_mullo_epi16(out, out);
    out = _mm256_slli_epi16(out, 7);
    _mm256_store_si256(output + i, out);
  }
  for (auto i = 0; i < half_chunks; ++i) {
    auto t_a = _mm256_load_si256(in_a + i);
    auto t_b = _mm256_load_si256(in_b + i);
    auto out = _mm256_mullo_epi16(t_a, t_b);
    out = _mm256_slli_epi16(out, 7);
    _mm256_store_si256(output + i, out);
  }
#endif

#ifdef BASE

#endif // DEBUG
}

inline void flatten_and_activate(int16_t *weights, int16_t *input,
                                 const int input_size) {
  // to be continued
}

// sum of all the 32 bit integers
inline int h_add(__m256i &value) {
  __m128i xmm0 = _mm256_castsi256_si128(value);
  __m128i xmm1 = _mm256_extracti128_si256(value, 1);
  xmm0 = _mm_add_epi32(xmm0, xmm1);
  xmm1 = _mm_unpackhi_epi64(xmm0, xmm0);
  xmm0 = _mm_add_epi32(xmm0, xmm1);
  xmm1 = _mm_shuffle_epi32(xmm0, _MM_SHUFFLE(2, 3, 0, 1));
  xmm0 = _mm_add_epi32(xmm0, xmm1);
  return _mm_cvtsi128_si32(xmm0);
}

inline void add_epi(__m256i &acc, __m256i a, __m256i b) {
  __m256i one = _mm256_set1_epi16(1);
  __m256i product0 = _mm256_madd_epi16(product0, one);

  // Add to the main int32 accumulator.
  acc = _mm256_add_epi32(acc, product0);
}
// below will be removed
inline int mull_accu_add(const int8_t *input, const int8_t *weights,
                         const int output_dim) {
  auto chunks = output_dim / 32;
  __m256i accu = _mm256_setzero_si256();
  for (int i = 0; i < output_dim; i += 16) {
    auto in = reinterpret_cast<const __m256i *>(input + i);
    auto in_weights = reinterpret_cast<const __m256i *>(weights + i);
    __m256i a = _mm256_stream_load_si256(in);
    __m256i b = _mm256_stream_load_si256(in_weights);
    add_epi(accu, a, b);
  }
  return h_add(accu);
}

/*
void test(int length) {
  const int ALIGNMENT = 32;

  auto weights =
      (int16_t *)std::aligned_alloc(ALIGNMENT, sizeof(int16_t) * length);
  auto input =
      (int16_t *)std::aligned_alloc(ALIGNMENT, sizeof(int16_t) * length);

  const int half_input = length / 2;
  for (auto i = 0; i < half_input; ++i) {
    weights[i] = i;
    weights[i + half_input] = i;
  }
  auto t1 = std::chrono::high_resolution_clock::now();
  accum_activation(weights, length);
  auto t2 = std::chrono::high_resolution_clock::now();
  auto dur =
      std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  // OUTPUT
  //
  std::cout << "Computation took us: " << dur << " ms" << std::endl;
  for (auto i = 0; i < half_input; ++i) {
    std::cout << weights[i] << std::endl;
  }
  free(weights);
  free(input);
}
int main(const int argl, const char **argc) {
  std::cout << "Hello world" << std::endl;
  int input = atoi(argc[1]);

  test(input);
}
*/
