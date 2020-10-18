//
// Created by Robin on 06.05.2018.
//
#include "Zobrist.h"


namespace Zobrist {
    std::array<std::array<uint64_t, 4>, 32> ZOBRIST_KEYS;
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
            uint32_t index = Bits::bitscan_foward(allPieces);
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
        const int pawn_index = (pos.getColor() == BLACK) ? BPAWN : WPAWN;
        const int king_index = (pos.getColor() == BLACK) ? BKING : WKING;
        const int opp_king_index = (pos.getColor() == BLACK) ? WKING : BKING;
        const int opp_pawn_index = (pos.getColor() == BLACK) ? WPAWN : BPAWN;
        auto toIndex = move.getToIndex();
        auto fromIndex = move.getFromIndex();
        constexpr uint32_t promo_squares = PROMO_SQUARES_WHITE | PROMO_SQUARES_BLACK;
        if (((move.from & pos.K) == 0u) && (move.to & promo_squares) != 0u) {
            pos.key ^= Zobrist::ZOBRIST_KEYS[toIndex][king_index];
            pos.key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pawn_index];
        } else if (((move.from & pos.K) != 0u)) {
            pos.key ^= Zobrist::ZOBRIST_KEYS[toIndex][king_index];
            pos.key ^= Zobrist::ZOBRIST_KEYS[fromIndex][king_index];
        } else {
            pos.key ^= Zobrist::ZOBRIST_KEYS[toIndex][pawn_index];
            pos.key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pawn_index];
        }
        uint32_t captures = move.captures;
        while (captures) {
            const uint32_t index = Bits::bitscan_foward(captures);
            if (((1u << index) & move.captures & pos.K) != 0) {
                pos.key ^= Zobrist::ZOBRIST_KEYS[index][opp_king_index];
            } else {
                pos.key ^= Zobrist::ZOBRIST_KEYS[index][opp_pawn_index];
            }
            captures &= captures - 1u;
        }

        pos.key ^= Zobrist::colorBlack;

    }

    uint64_t get_move_key(Position &cur_pos, Move move) {
        uint64_t move_key = 0ull;
        const int pawn_index = (cur_pos.getColor() == BLACK) ? BPAWN : WPAWN;
        const int king_index = (cur_pos.getColor() == BLACK) ? BKING : WKING;
        const int opp_king_index = (cur_pos.getColor() == BLACK) ? WKING : BKING;
        const int opp_pawn_index = (cur_pos.getColor() == BLACK) ? WPAWN : BPAWN;
        auto toIndex = move.getToIndex();
        auto fromIndex = move.getFromIndex();
        constexpr uint32_t promo_squares = PROMO_SQUARES_WHITE | PROMO_SQUARES_BLACK;
        if (((move.from & cur_pos.K) == 0u) && (move.to & promo_squares) != 0u) {
            move_key ^= Zobrist::ZOBRIST_KEYS[toIndex][king_index];
            move_key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pawn_index];
        } else if (((move.from & cur_pos.K) != 0u)) {
            move_key ^= Zobrist::ZOBRIST_KEYS[toIndex][king_index];
            move_key ^= Zobrist::ZOBRIST_KEYS[fromIndex][king_index];
        } else {
            move_key ^= Zobrist::ZOBRIST_KEYS[toIndex][pawn_index];
            move_key ^= Zobrist::ZOBRIST_KEYS[fromIndex][pawn_index];
        }
        uint32_t captures = move.captures;
        while (captures) {
            const uint32_t index = Bits::bitscan_foward(captures);
            if (((1u << index) & move.captures & cur_pos.K) != 0) {
                move_key ^= Zobrist::ZOBRIST_KEYS[index][opp_king_index];
            } else {
                move_key ^= Zobrist::ZOBRIST_KEYS[index][opp_pawn_index];
            }
            captures &= captures - 1u;
        }
        return move_key;
    }

}
