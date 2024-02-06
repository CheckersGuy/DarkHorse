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

enum class IndexType { INNER, PROMO };

inline uint32_t bitscan_foward(uint32_t bits) {

#ifndef _WIN32
  return __tzcnt_u32(bits);
#else
  return _tzcnt_u32(bits);
#endif
}

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
