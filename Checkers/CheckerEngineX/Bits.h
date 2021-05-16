//
// Created by root on 18.11.19.
//

#ifndef CHECKERENGINEX_BITS_H
#define CHECKERENGINEX_BITS_H

#include <cstdint>
#include <array>
#include <immintrin.h>


namespace Bits {
    uint32_t bitscan_foward(uint32_t bits);

    uint32_t pop_count(uint32_t val);

    uint32_t pext(uint32_t source, uint32_t mask);

}


#endif //CHECKERENGINEX_BITS_H
