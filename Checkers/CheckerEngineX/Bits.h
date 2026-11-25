//
// Created by root on 18.11.19.
//

#ifndef CHECKERENGINEX_BITS_H
#define CHECKERENGINEX_BITS_H

#include <array>
#include <cstdint>
#include <immintrin.h>
#include <types.h>

namespace Bits {
// reverses the bit order of a 32-bit word (= rotating the board by 180 degrees)
inline constexpr uint32_t reverse32(uint32_t x) {
  x = __builtin_bswap32(x);
  x = ((x & 0x0F0F0F0Fu) << 4) | ((x >> 4) & 0x0F0F0F0Fu);
  x = ((x & 0x33333333u) << 2) | ((x >> 2) & 0x33333333u);
  x = ((x & 0x55555555u) << 1) | ((x >> 1) & 0x55555555u);
  return x;
}


enum class IndexType { INNER, PROMO };

inline uint32_t bitscan_foward(uint32_t bits) { return __builtin_ctz(bits); }

inline uint32_t pop_count(uint32_t val) {
#ifndef _WIN32
  return __builtin_popcountll(val);
#else
  return __popcntq(val);
#endif
}

inline uint32_t pext(uint32_t source, uint32_t mask) {
#ifndef _WIN32
  return _pext_u32(source, mask);
#else
  return _pext_u32(source, mask);
#endif
}

inline auto leading_zero_count(uint32_t bits) { return __builtin_clz(bits); }


} // namespace Bits

#endif // CHECKERENGINEX_BITS_H
