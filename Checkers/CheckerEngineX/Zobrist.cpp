//
// Created by Robin on 06.05.2018.
//
#include "Zobrist.h"


namespace Zobrist {
    std::array<std::array<uint64_t ,4>,32> ZOBRIST_KEYS;
    uint64_t colorBlack = 0ull;


    uint64_t rand64() {

        static uint64_t seed = 1070372ull;
        seed ^= seed >> 12u;
        seed ^= seed << 25u;
        seed ^= seed >> 27u;
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

    void doUpdateZobristKey(Position &pos, Move move) {
        auto toIndex=move.getToIndex();
        auto fromIndex=move.getFromIndex();
        if (((move.getFrom() & pos.K) ==0u) && (move.getTo() & (PROMO_SQUARES_WHITE | PROMO_SQUARES_BLACK)) !=0 ) {
            pos.key ^= Zobrist::ZOBRIST_KEYS[toIndex][pos.getColor()*WKING];
            pos.key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pos.getColor()*WPAWN];
        } else if (((move.getFrom() & pos.K) !=0) ) {
            pos.key ^= Zobrist::ZOBRIST_KEYS[toIndex][pos.getColor()*WKING];
            pos.key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pos.getColor()*WKING];
        } else {
            pos.key ^= Zobrist::ZOBRIST_KEYS[toIndex][pos.getColor()*WPAWN];
            pos.key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pos.getColor()*WPAWN];
        }
        uint32_t captures = move.captures;
        while (captures) {
            const uint32_t index = __tzcnt_u32(captures);
            captures &= captures - 1u;
            if (((1u << index) & move.captures & pos.K) != 0) {
                pos.key ^= Zobrist::ZOBRIST_KEYS[index][pos.getColor()*BKING];
            } else {
                pos.key ^= Zobrist::ZOBRIST_KEYS[index][pos.getColor()*BPAWN];
            }
        }

        pos.key ^= Zobrist::colorBlack;
    }

}
