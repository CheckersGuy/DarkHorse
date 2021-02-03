//
// Created by root on 18.11.19.
//

#include "Bits.h"
namespace Bits {

#ifdef __EMSCRIPTEN__

    std::array<uint32_t, 255> bitScanArray;

    void set_up_bitscan() {

        for (uint32_t i = 0u; i < 255u; ++i) {
            uint32_t counter = 0u;
            uint32_t bitfield = i;
            while (bitfield) {
                counter++;
                bitfield = bitfield >> 1u;
            }
            bitScanArray[i] = counter - 1u;
        }
    }


#endif

    uint32_t bitscan_foward(uint32_t bits) {
#ifdef __EMSCRIPTEN__
        if ((bits & 0xff) != 0u)
            return bitScanArray[bits & 0xff];

        if (((bits >> 8u) & 0xff) != 0u)
            return bitScanArray[(bits >> 8u) & 0xff] + 8u;

        if (((bits >> 16u) & 0xff) != 0u)
            return bitScanArray[(bits >> 16u) & 0xff] + 16u;

        if (((bits >> 24u) & 0xff) != 0u)
            return bitScanArray[(bits >> 24u) & 0xff] + 24u;

        return 0u;

#else
        return __tzcnt_u32(bits);
#endif
    }

    uint32_t pop_count(uint32_t val) {
        return __builtin_popcount(val);
    }


    uint32_t pext(uint32_t source, uint32_t mask) {
        return _pext_u32(source,mask);
    }

}