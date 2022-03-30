/**///
// Created by Robin on 14.01.2018.
//


#ifndef CHECKERENGINEX_POSITION_H
#define CHECKERENGINEX_POSITION_H

#include "Move.h"
#include "types.h"
#include <sstream>
#include <optional>
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

    template<Color color, PieceType type>
    inline uint32_t get_pieces() const {
        if constexpr(color == BLACK && type == KING) {
            return BP & K;
        }
        if constexpr(color == WHITE && type == KING) {
            return WP & K;
        }

        if constexpr(color == BLACK && type == PAWN) {
            return BP & (~K);
        }
        if constexpr(color == WHITE && type == PAWN) {
            return WP & (~K);
        }
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

    std::string get_fen_string() const;

    Color get_color() const;

    template<Color color>
    bool has_jumps() const {
        return get_jumpers<color>() != 0;
    }

    bool has_jumps(Color color) const;

    bool has_jumps() const;

    bool is_wipe() const;

    bool has_threat() const;

    bool is_empty() const;

    bool is_end() const;

    bool is_legal() const;

    void make_move(Move move);


    void unmake_move(Move move);

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

    //given two consecutive positions,returns the move made
    static std::optional<Move> get_move(Position orig, Position next);

};

#endif //CHECKERENGINEX_POSITION_H
