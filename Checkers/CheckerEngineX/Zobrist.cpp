//
// Created by Robin on 06.05.2018.
//
#include "Zobrist.h"


namespace Zobrist {
    uint64_t ZOBRIST_KEYS[32][4];
    uint64_t colorBlack = 0;


    uint64_t rand64() {

        static uint64_t seed = 1070372ull;
        seed ^= seed >> 12;
        seed ^= seed << 25;
        seed ^= seed >> 27;
        return seed * 2685821657736338717ull;
    }


    void initializeZobrisKeys() {
        for (int i = 0; i < 32; ++i) {
            for (int j = 0; j < 4; ++j) {
                ZOBRIST_KEYS[i][j] = rand64();
            }
        }
        colorBlack = rand64();
    }

    uint64_t generateKey(const Position &pos) {
        const uint32_t BK = pos.K & pos.BP;
        const uint32_t WK = pos.K & pos.WP;
        uint64_t nextKey = 0u;
        uint32_t allPieces = pos.BP | pos.WP;
        while (allPieces) {
            uint32_t index = __tzcnt_u32(allPieces);
            uint32_t maske = 1u << index;
            if ((maske & BK)) {
                nextKey ^= ZOBRIST_KEYS[index][BKING];
            } else if ((maske & pos.BP)) {
                nextKey ^= ZOBRIST_KEYS[index][BPAWN];
            }
            if ((maske & WK)) {
                nextKey ^= ZOBRIST_KEYS[index][WKING];
            } else if ((maske & pos.WP)) {
                nextKey ^= ZOBRIST_KEYS[index][WPAWN];
            }
            allPieces &= allPieces - 1u;
        }
        if (pos.color == BLACK) {
            nextKey ^= colorBlack;
        }
        return nextKey;
    }

}
