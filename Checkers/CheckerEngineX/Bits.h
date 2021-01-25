//
// Created by root on 18.11.19.
//

#ifndef CHECKERENGINEX_BITS_H
#define CHECKERENGINEX_BITS_H

#include <cstdint>
#include <array>

#ifndef __EMSCRIPTEN__
#include <immintrin.h>
#endif

namespace Bits {
#ifdef __EMSCRIPTEN__


    void set_up_bitscan();

#endif

    uint32_t bitscan_foward(uint32_t bits);



    uint32_t pop_count(uint32_t val);

}


#endif //CHECKERENGINEX_BITS_H
