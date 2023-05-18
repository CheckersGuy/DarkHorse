// moresimd stuff
#include <bits/chrono.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <new>

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

  auto weights = (int8_t *)std::aligned_alloc(ALIGNMENT, length);
  auto input = (int8_t *)std::aligned_alloc(ALIGNMENT, length);

  for (auto i = 0; i < length; ++i) {
    input[i] = (i % 128);
    weights[i] = (i % 128);
  }

  auto t1 = std::chrono::high_resolution_clock::now();
    int sum = 0;
    for (auto i = 0; i < length; i++) {
      sum += input[i] * weights[i];
    }
__m256i accu = _mm256_setzero_si256();

for (auto i = 0; i < length; i += 32) {
  auto in_a = reinterpret_cast<const __m256i *>(weights + i);
  auto in_b = reinterpret_cast<const __m256i *>(input + i);

  __m256i a = _mm256_stream_load_si256(in_a);
  __m256i b = _mm256_stream_load_si256(in_b);
  add_epi(accu, a, b);
}
int sum = h_add(accu);
int sum = mull_accu_add(input, weights, length);
auto t2 = std::chrono::high_resolution_clock::now();
auto elapsed =
    std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count();
std::cout << "Time : " << elapsed << std::endl;
std::cout << "The sum was: " << sum << std::endl;
free(weights);
free(input);
}
*/
/*
int main(const int argl, const char **argc) {
  std::cout << "Hello world" << std::endl;
  int input = atoi(argc[1]);
  int temp = input;

  test(temp);
}
*/
