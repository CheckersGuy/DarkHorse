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

    uint64_t generateKey(const Position &pos, Color color) {
        const uint32_t BK = pos.K & pos.BP;
        const uint32_t WK = pos.K & pos.WP;
        uint64_t nextKey = 0;
        for (int i = 0; i < 32; ++i) {
            const uint32_t maske = 1 << i;
            if ((maske & BK) == maske) {
                nextKey ^= ZOBRIST_KEYS[i][BKING];
            } else if ((maske & pos.BP) == maske) {
                nextKey ^= ZOBRIST_KEYS[i][BPAWN];
            }

            if ((maske & WK) == maske) {
                nextKey ^= ZOBRIST_KEYS[i][WKING];
            } else if ((maske & pos.WP) == maske) {
                nextKey ^= ZOBRIST_KEYS[i][WPAWN];
            }
        }
        if (color == BLACK) {
            nextKey ^= colorBlack;
        }
        return nextKey;
    }

}

void Zobrist::doUpdateZobristKey(Position &pos, Move move) {
    if (pos.color == WHITE) {
        if (move.getPieceType() == 0 && (move.getTo() >> 2) == 0) {
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getTo()][WKING];
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getFrom()][WPAWN];
        } else if (move.getPieceType() == 1) {
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getTo()][WKING];
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getFrom()][WKING];
        } else {
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getTo()][WPAWN];
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getFrom()][WPAWN];
        }
        uint32_t captures = move.captures;
        while (captures) {
            const uint32_t index = __tzcnt_u32(captures);
            captures = __blsr_u32(captures);
            if (((1 << index) & move.captures & pos.K) != 0) {
                pos.key ^= Zobrist::ZOBRIST_KEYS[index][BKING];
            } else {
                pos.key ^= Zobrist::ZOBRIST_KEYS[index][BPAWN];
            }
        }

    } else {
        if (move.getPieceType() == 0 && (move.getTo() >> 2) == 7) {
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getTo()][BKING];
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getFrom()][BPAWN];
        } else if (move.getPieceType() == 1) {
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getTo()][BKING];
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getFrom()][BKING];
        } else {
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getTo()][BPAWN];
            pos.key ^= Zobrist::ZOBRIST_KEYS[move.getFrom()][BPAWN];
        }
        uint32_t captures = move.captures;
        while (captures) {
            const uint32_t index = __tzcnt_u32(captures);
            captures &= captures - 1;
            if (((1 << index) & move.captures & pos.K) != 0) {
                pos.key ^= Zobrist::ZOBRIST_KEYS[index][WKING];
            } else {
                pos.key ^= Zobrist::ZOBRIST_KEYS[index][WPAWN];
            }
        }
    }
    pos.key ^= Zobrist::colorBlack;
}
