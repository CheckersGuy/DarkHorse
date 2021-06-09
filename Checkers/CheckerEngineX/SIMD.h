//
// Created by root on 01.06.21.
//

#ifndef READING_SIMD_H
#define READING_SIMD_H

#include <immintrin.h>
//some helper functions

inline float hsum_ps_sse1(__m128 v) {                                  // v = [ D C | B A ]
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
    __m128 sums = _mm_add_ps(v, shuf);      // sums = [ D+C C+D | B+A A+B ]
    shuf = _mm_movehl_ps(shuf, sums);      //  [   C   D | D+C C+D ]  // let the compiler avoid a mov by reusing shuf
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

inline float dot_product_simd(float *a, float *b, size_t nums) {
    //does element wise multiplication
    __m128 m = _mm_set1_ps(0.0);
    for (auto i = 0; i < nums; i += 4) {
        __m128 val_a = _mm_load_ps(a + i);
        __m128 val_b = _mm_load_ps(b + i);
        __m128 result = _mm_mul_ps(val_a, val_b);
        m = _mm_add_ps(result, m);
    }
    return hsum_ps_sse1(m);
}

inline void add_simd(float *a, float *b, float *c, float mul, size_t nums) {
    //computes a +b*mul and stores result in c
    __m256 t = _mm256_set1_ps(mul);
    for (auto i = 0; i < nums; i += 8) {
        __m256 t1 = _mm256_loadu_ps(a + i);
        __m256 t2 = _mm256_loadu_ps(b + i);
        t2 = _mm256_mul_ps(t2, t);
        __m256 result = _mm256_add_ps(t2, t1);
        _mm256_store_ps(c + i, result);
    }
}

inline void add_bias_and_clamp_simd(float *input, float *bias, size_t nums) {
    __m256 zeros = _mm256_set1_ps(0.0f);
    __m256 ones = _mm256_set1_ps(1.0f);
    for (auto i = 0; i < nums; i += 8) {
        __m256 v = _mm256_load_ps(input + i);
        __m256 v2 = _mm256_load_ps(bias + i);
        v = _mm256_add_ps(v, v2);
        v = _mm256_max_ps(v, zeros);
        v = _mm256_min_ps(v, ones);
        _mm256_store_ps(input + i, v);
    }
}


#endif //READING_SIMD_H
