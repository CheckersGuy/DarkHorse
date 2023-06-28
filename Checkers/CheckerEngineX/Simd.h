#include <algorithm>
#include <bits/chrono.h>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <emmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <new>
#include <random>
#define AVX256

namespace Simd {

inline int32_t hsum_epi32_avx(__m128i x) {
  __m128i hi64 = _mm_unpackhi_epi64(x, x);
  __m128i sum64 = _mm_add_epi32(hi64, x);
  __m128i hi32 = _mm_shuffle_epi32(sum64, _MM_SHUFFLE(2, 3, 0, 1));
  __m128i sum32 = _mm_add_epi32(sum64, hi32);
  return _mm_cvtsi128_si32(sum32);
}

inline int32_t hsum_8x32(__m256i v) {
  __m128i sum128 =
      _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  return hsum_epi32_avx(sum128);
}

template <int input_size>
inline void accum_activation8(int16_t *acc, uint8_t *out) {

#ifdef AVX256

  constexpr int num_chunks = input_size / 16;
  constexpr int control = 0b11011000;

  auto *in_a = reinterpret_cast<const __m256i *>(acc);
  auto *output = reinterpret_cast<__m256i *>(out);
  const auto max_val = _mm256_set1_epi16(127);
  const auto min_val = _mm256_set1_epi16(0);
  for (auto i = 0; i < num_chunks / 4; ++i) {
    auto temp0 = _mm256_load_si256(in_a + 2 * i);
    auto temp1 = _mm256_load_si256(in_a + num_chunks / 2 + 2 * i);

    auto temp2 = _mm256_load_si256(in_a + 2 * i + 1);
    auto temp3 = _mm256_load_si256(in_a + num_chunks / 2 + 2 * i + 1);

    temp0 = _mm256_max_epi16(temp0, min_val);
    temp0 = _mm256_min_epi16(temp0, max_val);
    temp0 = _mm256_mullo_epi16(temp0, temp0);
    temp0 = _mm256_srai_epi16(temp0, 7);

    temp1 = _mm256_max_epi16(temp1, min_val);
    temp1 = _mm256_min_epi16(temp1, max_val);
    temp1 = _mm256_mullo_epi16(temp1, temp1);
    temp1 = _mm256_srai_epi16(temp1, 7);

    temp2 = _mm256_max_epi16(temp2, min_val);
    temp2 = _mm256_min_epi16(temp2, max_val);
    temp2 = _mm256_mullo_epi16(temp2, temp2);
    temp2 = _mm256_srai_epi16(temp2, 7);

    temp3 = _mm256_max_epi16(temp3, min_val);
    temp3 = _mm256_min_epi16(temp3, max_val);
    temp3 = _mm256_mullo_epi16(temp3, temp3);
    temp3 = _mm256_srai_epi16(temp3, 7);

    auto result0 = _mm256_srai_epi16(_mm256_mullo_epi16(temp0, temp1), 7);
    auto result1 = _mm256_srai_epi16(_mm256_mullo_epi16(temp2, temp3), 7);
    auto packed =
        _mm256_permute4x64_epi64(_mm256_packs_epi16(result0, result1), control);

    _mm256_store_si256(output + i, packed);
  }
#endif

#ifdef BASE

  for (auto i = 0; i < input_size / 2; ++i) {
    int16_t val = acc[i];
    val = std::clamp(val, int16_t{0}, int16_t{127});
    val = (val * val) / 128;

    int16_t val2 = acc[i + input_size / 2];
    val2 = std::clamp(val2, int16_t{0}, int16_t{127});
    val2 = (val2 * val2) / 128;

    out[i] = (val * val2) / 128;
  }

#endif // DEBUG
}

template <int length>
inline void square_clipped8(int32_t *input, uint8_t *output) {

#ifdef AVX256
  constexpr int num_chunks = length / 8;
  auto *in = reinterpret_cast<const __m256i *>(input);
  auto *out = reinterpret_cast<__m256i *>(output);
  const auto max_val = _mm256_set1_epi16(127);
  const auto min_val = _mm256_set1_epi16(0);
  const __m256i control = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);
  for (auto i = 0; i < num_chunks / 4; ++i) {
    auto temp1 = _mm256_load_si256(in + 4 * i);
    auto temp2 = _mm256_load_si256(in + 4 * i + 1);
    auto temp3 = _mm256_load_si256(in + 4 * i + 2);
    auto temp4 = _mm256_load_si256(in + 4 * i + 3);

    auto packed1 = _mm256_packs_epi32(temp1, temp2);
    auto packed2 = _mm256_packs_epi32(temp3, temp4);

    packed1 = _mm256_max_epi16(packed1, min_val);
    packed1 = _mm256_min_epi16(packed1, max_val);
    packed1 = _mm256_mullo_epi16(packed1, packed1);
    packed1 = _mm256_srai_epi16(packed1, 7);

    packed2 = _mm256_max_epi16(packed2, min_val);
    packed2 = _mm256_min_epi16(packed2, max_val);
    packed2 = _mm256_mullo_epi16(packed2, packed2);
    packed2 = _mm256_srai_epi16(packed2, 7);

    auto result = _mm256_permutevar8x32_epi32(
        _mm256_packs_epi16(packed1, packed2), control);
    _mm256_store_si256(out + i, result);
  }

#endif

#ifdef BASE
  for (auto i = 0; i < length; ++i) {
    int val = std::clamp(input[i], 0, 127);
    val = val * val;
    val = val / 128;
    output[i] = val;
  }
#endif
}

template <int length> inline void vec_add(const int16_t *input, int16_t *out) {
#ifdef AVX256
  // some experimental stuff
  constexpr int num_chunks = length / 16;
  auto accu = reinterpret_cast<__m256i *>(out);
  auto in = reinterpret_cast<const __m256i *>(input);
  for (auto i = 0; i < num_chunks; ++i) {
    auto a = _mm256_load_si256(in + i);
    auto b = _mm256_load_si256(accu + i);
    auto res = _mm256_add_epi16(a, b);
    _mm256_store_si256(accu + i, res);
  }

#endif

#ifdef BASE
  for (auto i = 0; i < length; ++i) {
    out[i] += input[i];
  }

#endif
}
template <int length> inline void vec_diff(const int16_t *input, int16_t *out) {
#ifdef AVX256
  constexpr int num_chunks = length / 16;
  auto accu = reinterpret_cast<__m256i *>(out);
  auto in = reinterpret_cast<const __m256i *>(input);
  for (auto i = 0; i < num_chunks; ++i) {
    auto a = _mm256_load_si256(in + i);
    auto b = _mm256_load_si256(accu + i);
    auto res = _mm256_sub_epi16(b, a);
    _mm256_store_si256(accu + i, res);
  }

#endif

#ifdef BASE
  for (auto i = 0; i < length; ++i) {
    out[i] -= input[i];
  }

#endif
}

inline int m256_hadd(__m256i sum) {
  __m128i sum128 = _mm_add_epi32(_mm256_castsi256_si128(sum),
                                 _mm256_extracti128_si256(sum, 1));
  sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_BADC));
  sum128 = _mm_add_epi32(sum128, _mm_shuffle_epi32(sum128, _MM_PERM_CDAB));
  return _mm_cvtsi128_si32(sum128);
}
template <int length>
inline int flatten8(const int8_t *weights, const uint8_t *input) {
#ifdef AVX256
  __m256i acc = _mm256_setzero_si256();

  constexpr int num_chunks = length / 32;
  auto *w = reinterpret_cast<const __m256i *>(weights);
  auto *in = reinterpret_cast<const __m256i *>(input);
  for (auto i = 0; i < num_chunks; ++i) {
    auto w_temp = _mm256_load_si256(w + i);
    auto in_temp = _mm256_load_si256(in + i);

    auto temp = _mm256_maddubs_epi16(in_temp, w_temp);
    __m256i one = _mm256_set1_epi16(1);
    auto product = _mm256_madd_epi16(one, temp);
    acc = _mm256_add_epi32(acc, product);
  }
  return m256_hadd(acc);
#endif
#ifdef BASE
  int sum = 0;
  for (auto i = 0; i < length; ++i) {
    sum += input[i] * weights[i];
  }

  return sum;

#endif
}

inline __m128i m256_haddx4(__m256i sum0, __m256i sum1, __m256i sum2,
                           __m256i sum3, __m128i bias) {
  sum0 = _mm256_hadd_epi32(sum0, sum1);
  sum2 = _mm256_hadd_epi32(sum2, sum3);

  sum0 = _mm256_hadd_epi32(sum0, sum2);

  __m128i sum128lo = _mm256_castsi256_si128(sum0);
  __m128i sum128hi = _mm256_extracti128_si256(sum0, 1);

  return _mm_add_epi32(_mm_add_epi32(sum128lo, sum128hi), bias);
};
inline void m256_add_dpbusd_epi32(__m256i &acc, __m256i a, __m256i b) {
  __m256i product0 = _mm256_maddubs_epi16(a, b);

  __m256i one = _mm256_set1_epi16(1);
  product0 = _mm256_madd_epi16(product0, one);
  acc = _mm256_add_epi32(acc, product0);
}

} // namespace Simd
