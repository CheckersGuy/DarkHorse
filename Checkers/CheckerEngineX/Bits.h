//
// Created by root on 18.11.19.
//

#ifndef CHECKERENGINEX_BITS_H
#define CHECKERENGINEX_BITS_H

#include <cstdint>
#include <array>
#include <immintrin.h>


namespace Bits {
    inline uint32_t bitscan_foward(uint32_t bits) {

#ifndef _WIN32
        return __tzcnt_u32(bits);
#else
        return _tzcnt_u32(bits);
#endif
    }

    inline uint32_t pop_count(uint32_t val) {
#ifndef _WIN32
        return __builtin_popcount(val);
#else
        return _popcount(val)
#endif
    }


    inline uint32_t pext(uint32_t source, uint32_t mask) {
#ifndef _WIN32
        return _pext_u32(source, mask);
#else
        return _pext_u32(source, mask);
#endif
    }

}


#endif //CHECKERENGINEX_BITS_H
