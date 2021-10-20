//
// Created by root on 18.11.19.
//

#ifndef CHECKERENGINEX_BITS_H
#define CHECKERENGINEX_BITS_H

#include <cstdint>
#include <array>
#include <immintrin.h>
#include <types.h>

namespace Bits {

    const size_t offset1 = 8ull * 157464ull;
    const size_t offset2 = 4ull * 531441ull + 8ull * 157464ull;

    enum class IndexType {
        INNER, PROMO
    };


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
        return __popcnt(val);
#endif
    }


    inline uint32_t pext(uint32_t source, uint32_t mask) {
#ifndef _WIN32
        return _pext_u32(source, mask);
#else
        return _pext_u32(source, mask);
#endif
    }

    template<IndexType type>
    size_t get_big_index(uint32_t reg, uint32_t wp, uint32_t bp, uint32_t k) {
        const uint32_t PROMO_SQUARES = PROMO_SQUARES_BLACK | PROMO_SQUARES_WHITE;
        size_t index = 0ull;
        uint32_t BP = bp & (~k);
        uint32_t WP = wp & (~k);
        uint32_t orig_pieces = (bp | wp) & reg & (~PROMO_SQUARES);
        uint32_t promo_region = reg & PROMO_SQUARES;

        uint32_t promo_pieces = (bp | wp) & PROMO_SQUARES & reg;
        uint32_t promo_index = Bits::pext(promo_pieces, promo_region);
        uint32_t pieces = Bits::pext(orig_pieces, reg & (~PROMO_SQUARES));

        while (pieces) {
            uint32_t lsb = (orig_pieces & ~(orig_pieces - 1u));
            size_t temp_index = Bits::bitscan_foward(pieces);
            size_t current = ((BP & lsb) != 0u) * 1ull + ((WP & lsb) != 0u) * 2ull;
            index += current * powers3[temp_index];
            pieces &= pieces - 1u;
            orig_pieces &= orig_pieces - 1u;
        }

        if constexpr(type == IndexType::INNER) {
            return index;
        } else {
            return 8 * index + promo_index;
        }
    }


    inline size_t getIndex2(uint32_t reg, uint32_t wp, uint32_t bp, uint32_t k) {
        uint32_t orig_pieces = (bp | wp) & reg;
        uint32_t pieces = (bp | wp);
        pieces = Bits::pext(pieces, reg);

        uint32_t BP = bp & (~k);
        uint32_t WP = wp & (~k);
        uint32_t BK = bp & k;
        uint32_t WK = wp & k;
        size_t index = 0ull;
        while (orig_pieces) {
            uint32_t lsb = (orig_pieces & ~(orig_pieces - 1u));
            size_t temp_index = Bits::bitscan_foward(pieces);
            size_t current = ((BP & lsb) != 0u) * 1ull + ((WP & lsb) != 0u) * 2ull + ((BK & lsb) != 0u) * 3ull +
                             ((WK & lsb) != 0u) * 4ull;

            index += current * powers5[temp_index];
            pieces &= pieces - 1u;
            orig_pieces &= orig_pieces - 1u;
        }

        return index;
    }


    template<typename Callback>
    void big_index(Callback func, uint32_t WP, uint32_t BP, uint32_t K) {
        for (auto i: {0, 2}) {
            size_t temp = ((i == 0) ? 0 : 1);
            for (auto k = 0; k < 2; ++k) {
                const uint32_t sub_reg = big_region << (8 * i + k);
                size_t index = get_big_index<IndexType::PROMO>(sub_reg, WP, BP, K);
                size_t sub_index = 8 * index + 2 * k + 4 * temp;
                func(sub_index);
            }
        }
        //FOR THE NON_PROMO_SQUARES
        for (auto k = 0; k < 2; ++k) {
            const uint32_t sub_reg = big_region << (8 * 1 + k);
            size_t index = get_big_index<IndexType::INNER>(sub_reg, WP, BP, K);
            size_t sub_index = 4 * index + 2 * k + offset1;
            func(sub_index);
        }
    }

    template<typename Callback>
    void small_index(Callback func, uint32_t WP, uint32_t BP, uint32_t K) {
        for (auto i = 0; i < 3; ++i) {
            for (auto k = 0; k < 3; ++k) {
                const uint32_t sub_reg = region << (8 * i + k);
                size_t index = getIndex2(sub_reg, WP, BP, K);
                size_t sub_index = 18 * index + 2 * k + 6 * i + offset2;
                func(sub_index);
            }
        }
    }

}

#endif //CHECKERENGINEX_BITS_H
