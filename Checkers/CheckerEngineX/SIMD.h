//
// Created by root on 01.06.21.
//

#ifndef READING_SIMD_H
#define READING_SIMD_H

#include <immintrin.h>
//some helper functions

float hsum_ps_sse1(__m128 v) {                                  // v = [ D C | B A ]
    __m128 shuf = _mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 3, 0, 1));  // [ C D | A B ]
    __m128 sums = _mm_add_ps(v, shuf);      // sums = [ D+C C+D | B+A A+B ]
    shuf = _mm_movehl_ps(shuf, sums);      //  [   C   D | D+C C+D ]  // let the compiler avoid a mov by reusing shuf
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

float dot_product_simd(float *a, float *b, size_t nums) {
    //does element wise multiplication
    __m128 m = _mm_set1_ps(0.0);
    for (auto i = size_t{0}; i < nums; i += 4u) {
        __m128 val_a = _mm_load_ps(a + i);
        __m128 val_b = _mm_load_ps(b + i);
        __m128 result = _mm_mul_ps(val_a, val_b);
        m = _mm_add_ps(result, m);
    }
    return hsum_ps_sse1(m);
}

#endif //READING_SIMD_H
