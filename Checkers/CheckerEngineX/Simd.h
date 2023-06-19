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
#define BASE

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
inline void accum_activation(int16_t *acc, int16_t *out) {
  // combined the element-wise multiplicaiton and the square clipped relu
  // maybe its a good idea to split this up at some point
#ifdef AVX256

  constexpr int num_chunks = input_size / 16;
  constexpr int half_chunks = num_chunks / 2;

  auto *in_a = reinterpret_cast<const __m256i *>(acc);
  auto *output = reinterpret_cast<__m256i *>(out);
  const auto max_val = _mm256_set1_epi16(127);
  const auto min_val = _mm256_set1_epi16(0);
  for (auto i = 0; i < half_chunks; ++i) {
    auto temp = _mm256_load_si256(in_a + i);
    auto temp2 = _mm256_load_si256(in_a + half_chunks + i);
    temp = _mm256_max_epi16(temp, min_val);
    temp = _mm256_min_epi16(temp, max_val);
    temp = _mm256_mullo_epi16(temp, temp);
    temp = _mm256_srai_epi16(temp, 7);

    temp2 = _mm256_max_epi16(temp2, min_val);
    temp2 = _mm256_min_epi16(temp2, max_val);
    temp2 = _mm256_mullo_epi16(temp2, temp2);
    temp2 = _mm256_srai_epi16(temp2, 7);

    auto result = _mm256_srai_epi16(_mm256_mullo_epi16(temp, temp2), 7);
    _mm256_store_si256(output + i, result);
  }

#endif

#ifdef BASE

  for (auto i = 0; i < input_size; ++i) {
    int16_t val = acc[i];
    val = std::clamp(val, int16_t{0}, int16_t{127});
    out[i] = val;
  }

#endif // DEBUG
}
template <int input_size>
inline void accum_activation8(int16_t *acc, int8_t *out) {

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

  for (auto i = 0; i < input_size; ++i) {
    int16_t val = acc[i];
    val = std::clamp(val, int16_t{0}, int16_t{127});

    out[i] = val;
  }

#endif // DEBUG
}

inline void square_clipped(int32_t *input, int16_t *output, const int length) {
// to be continued
#ifdef AVX256
  const int control = 0b11011000;
  const int num_chunks = length / 8;
  auto *in = reinterpret_cast<const __m256i *>(input);
  auto *out = reinterpret_cast<__m256i *>(output);
  const auto max_val = _mm256_set1_epi16(127);
  const auto min_val = _mm256_set1_epi16(0);

  for (auto i = 0; i < num_chunks / 2; ++i) {
    auto temp = _mm256_load_si256(in + 2 * i);
    auto temp2 = _mm256_load_si256(in + 2 * i + 1);

    // 16 bit integers below
    auto packed =
        _mm256_permute4x64_epi64(_mm256_packs_epi32(temp, temp2), control);

    packed = _mm256_max_epi16(packed, min_val);
    packed = _mm256_min_epi16(packed, max_val);

    auto result = _mm256_srli_epi16(_mm256_mullo_epi16(packed, packed), 7);
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
template <int length>
inline void square_clipped8(int32_t *input, int8_t *output) {

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
    output[i] = val;
  }
#endif
}

inline void square_clipped8_128(int32_t *input, int8_t *output,
                                const int length) {
  // to be continued
#ifdef AVX256
  const int num_chunks = length / 4;
  auto *in = reinterpret_cast<const __m128i *>(input);
  auto *out = reinterpret_cast<__m128i *>(output);
  const auto max_val = _mm_set1_epi16(127);
  const auto min_val = _mm_set1_epi16(0);
  for (auto i = 0; i < num_chunks / 4; ++i) {
    auto temp1 = _mm_load_si128(in + 4 * i);
    auto temp2 = _mm_load_si128(in + 4 * i + 1);
    auto temp3 = _mm_load_si128(in + 4 * i + 2);
    auto temp4 = _mm_load_si128(in + 4 * i + 3);

    auto packed1 = _mm_packs_epi32(temp1, temp2);
    auto packed2 = _mm_packs_epi32(temp3, temp4);

    packed1 = _mm_max_epi16(packed1, min_val);
    packed1 = _mm_min_epi16(packed1, max_val);
    packed1 = _mm_mullo_epi16(packed1, packed1);
    packed1 = _mm_srai_epi16(packed1, 7);

    packed2 = _mm_max_epi16(packed2, min_val);
    packed2 = _mm_min_epi16(packed2, max_val);
    packed2 = _mm_mullo_epi16(packed2, packed2);
    packed2 = _mm_srai_epi16(packed2, 7);

    auto result = _mm_packs_epi16(packed1, packed2);
    _mm_store_si128(out + i, result);
  }

#endif

#ifdef BASE
  for (auto i = 0; i < length; ++i) {
    int val = std::clamp(input[i], 0, 127);
    output[i] = val;
  }

#endif
}

template <int length>
int flatten(const int16_t *weights, const int16_t *input) {
#ifdef AVX256
  __m256i acc = _mm256_setzero_si256();

  const int num_chunks = length / 16;
  auto *w = reinterpret_cast<const __m256i *>(weights);
  auto *in = reinterpret_cast<const __m256i *>(input);

  for (auto i = 0; i < num_chunks; ++i) {
    auto w_temp = _mm256_load_si256(w + i);
    auto in_temp = _mm256_load_si256(in + i);
    auto sum = _mm256_madd_epi16(w_temp, in_temp);
    acc = _mm256_add_epi32(acc, sum);
  }
  return hsum_8x32(acc);
#endif

#ifdef BASE
  int32_t sum = 0;
  for (auto i = 0; i < length; ++i) {
    sum += input[i] * weights[i];
  }
  return sum;

#endif
}
template <int length> inline void vec_add(const int16_t *input, int16_t *out) {
#ifdef AVX256
  // some experimental stuff
  const int num_chunks = length / 16;
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
  const int num_chunks = length / 16;
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
inline int flatten8(const int8_t *weights, const int8_t *input) {
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
inline int flatten8_128(const int8_t *weights, const int8_t *input,
                        const int length) {
#ifdef AVX256
  __m128i acc = _mm_setzero_si128();

  const int num_chunks = length / 16;
  auto *w = reinterpret_cast<const __m128i *>(weights);
  auto *in = reinterpret_cast<const __m128i *>(input);
  for (auto i = 0; i < num_chunks; ++i) {
    auto w_temp = _mm_load_si128(w + i);
    auto in_temp = _mm_load_si128(in + i);

    auto temp = _mm_maddubs_epi16(in_temp, w_temp);
    __m128i one = _mm_set1_epi16(1);
    auto product = _mm_madd_epi16(one, temp);
    acc = _mm_add_epi32(acc, product);
  }
  return hsum_epi32_avx(acc);
#endif
#ifdef BASE
  int sum = 0;
  for (auto i = 0; i < length; ++i) {
    sum += input[i] * weights[i];
  }

  return sum;

#endif
}

} // namespace Simd
/*
int main(const int argl, const char **argc) {
  std::cout << "Hello world" << std::endl;
  int input = atoi(argc[1]);

  Simd::test(input);
}
*/
