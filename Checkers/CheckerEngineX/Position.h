/**///
// Created by Robin on 14.01.2018.
//


#ifndef CHECKERENGINEX_POSITION_H
#define CHECKERENGINEX_POSITION_H

#include "Move.h"
#include "types.h"


const uint32_t temp_mask = 0xf;


inline constexpr uint32_t getHorizontalFlip(uint32_t b) {
    uint32_t x = ((b & MASK_COL_4)) >> 3u;
    x |= (b & MASK_COL_3) >> 1u;
    x |= (b & MASK_COL_1) << 3u;
    x |= (b & MASK_COL_2) << 1u;
    return x;
}

inline constexpr uint32_t getVerticalFlip(uint32_t b) {
    uint32_t x = b >> 28u;
    x |= (b >> 20u) & 0xf0u;
    x |= (b >> 12u) & 0xf00u;
    x |= (b >> 4u) & 0xf000u;

    x |= b << 28u;
    x |= (b << 20u) & 0x0f000000u;
    x |= (b << 12u) & 0x00f00000u;
    x |= (b << 4u) & 0x000f0000u;
    return x;
}


inline constexpr uint32_t getMirrored(uint32_t b) {
    return getHorizontalFlip(getVerticalFlip(b));
}

struct Square {
    PieceType type;
    uint32_t index;

    friend std::ostream &operator<<(std::ostream &stream, Square square);
};

struct Position {
    Color color{BLACK};
    uint32_t WP{0u}, BP{0u}, K{0u};
    uint64_t key{0ull};

    uint32_t piece_count();

    template<Color color>
    constexpr uint32_t get_current() const {
        if constexpr (color == BLACK)
            return BP;
        else
            return WP;
    }

    uint32_t getKingAttackSquares(uint32_t bit_mask);

    template<Color color>
    uint32_t attacks() const {
        //returns all empty squares that are attacked by color
        uint32_t attacks = 0u;
        const uint32_t empty = ~(BP | WP);
        auto pawns = get_current<color>(); //kings and pawns
        auto kings = get_current<color>() & K; //only kings
        attacks |= (defaultShift<color>(pawns) | forwardMask<color>(pawns));
        attacks |= (defaultShift<~color>(kings) | forwardMask<~color>(kings));
        attacks &= empty;
        return attacks;
    }

    template<Color color>
    uint32_t get_movers() const {
        const uint32_t nocc = ~(BP | WP);
        const uint32_t current = get_current<color>();
        const uint32_t kings = current & K;

        uint32_t movers = (defaultShift<~color>(nocc) | forwardMask<~color>(nocc)) & current;
        if (kings) {
            movers |= (defaultShift<color>(nocc) | forwardMask<color>(nocc)) & kings;
        }
        return movers;
    }

    template<Color color>
    uint32_t get_jumpers() const {
        const uint32_t nocc = ~(BP | WP);
        const uint32_t current = get_current<color>();
        const uint32_t opp = get_current<~color>();
        const uint32_t kings = current & K;

        uint32_t movers = 0u;
        uint32_t temp = defaultShift<~color>(nocc) & opp;
        if (temp != 0u) {
            movers |= forwardMask<~color>(temp) & current;
        }
        temp = forwardMask<~color>(nocc) & opp;
        if (temp != 0u) {
            movers |= defaultShift<~color>(temp) & current;
        }
        if (kings != 0u) {
            temp = defaultShift<color>(nocc) & opp;
            if (temp != 0u) {
                movers |= forwardMask<color>(temp) & kings;
            }
            temp = forwardMask<color>(nocc) & opp;

            if (temp != 0u) {
                movers |= defaultShift<color>(temp) & kings;
            }
        }
        return movers;
    }

    PieceType get_piece_type(Move move) const;


    std::string get_fen_string() const;

    Color get_color() const;

    bool has_jumps(Color color) const;

    bool has_jumps() const;

    bool is_wipe() const;

    bool has_threat() const;

    bool is_empty() const;

    bool is_end() const;

    bool is_legal() const;

    void make_move(Move &move);

    void make_move(uint32_t from_index, uint32_t to_index);

    void print_position() const;

    Position get_color_flip() const;

    static Position get_start_position();

    static Position pos_from_fen(std::string fen_string);

    inline bool operator==(const Position &pos) const {
        return (pos.BP == BP && pos.WP == WP && pos.K == K && pos.color == color);
    }

    inline bool operator!=(const Position &other) const {
        return !(*this == other);
    }

    friend std::ostream &operator<<(std::ostream &stream, const Position &pos);

    friend std::istream &operator>>(std::istream &stream, Position &pos);

};

#endif //CHECKERENGINEX_POSITION_H
