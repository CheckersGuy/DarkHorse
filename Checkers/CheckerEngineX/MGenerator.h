//
// Created by Robin on 11.05.2017.
//
#ifndef CHECKERSTEST_MGENERATOR_H
#define CHECKERSTEST_MGENERATOR_H

#include "Board.h"
#include "MoveListe.h"


template<Color color>
inline void mask_bits(Position &pos, const uint32_t maske) {
    if constexpr (color == BLACK)
        pos.BP ^= maske;
    else
        pos.WP ^= maske;
}


template<Color color>
inline
void get_silent_moves(const Position &pos, MoveListe &liste) {
    uint32_t movers = pos.get_movers<color>();
    const uint32_t nocc = ~(pos.BP | pos.WP);
    while (movers) {
        const uint32_t maske = movers & ~(movers - 1u);
        uint32_t squares = defaultShift<color>(maske) | forwardMask<color>(maske);
        squares |= forwardMask<~color>(maske & pos.K) | defaultShift<~color>(maske & pos.K);
        squares &= nocc;
        while (squares) {
            const uint32_t next = squares & ~(squares - 1u);;
            liste.add_move(Move{maske, next});
            squares &= squares - 1u;
        }
        movers &= movers - 1u;
    }
}

template<Color color, PieceType type>
inline
void
add_king_captures(const Position &pos, const uint32_t orig, const uint32_t current, const uint32_t captures,
                  MoveListe &liste) {
    const uint32_t opp = pos.get_current<~color>() ^ captures;
    const uint32_t nocc = ~(opp | pos.get_current<color>());
    const uint32_t temp0 = defaultShift<color>(current) & opp;
    const uint32_t temp1 = forwardMask<color>(current) & opp;
    const uint32_t dest0 = forwardMask<color>(temp0) & nocc;
    const uint32_t dest1 = defaultShift<color>(temp1) & nocc;

    uint32_t imed = (forwardMask<~color>(dest0) | defaultShift<~color>(dest1));
    uint32_t dest = dest0 | dest1;
    if constexpr (type == KING) {
        const uint32_t temp2 = defaultShift<~color>(current) & opp;
        const uint32_t temp3 = forwardMask<~color>(current) & opp;
        const uint32_t dest2 = forwardMask<~color>(temp2) & nocc;
        const uint32_t dest3 = defaultShift<~color>(temp3) & nocc;
        imed |= forwardMask<color>(dest2) | defaultShift<color>(dest3);
        dest |= dest2 | dest3;
    }
    if (dest == 0u) {
        liste.add_move(Move{orig, current, captures});
    }
    while (dest) {
        uint32_t destMask = dest & ~(dest - 1u);
        uint32_t capMask = imed & ~(imed - 1u);
        dest &= dest - 1u;
        imed &= imed - 1u;
        add_king_captures<color, type>(pos, orig, destMask, (captures | capMask), liste);
    }
}

template<Color color>
inline
void loop_captures(Position &pos, MoveListe &liste) {
    uint32_t movers = pos.get_jumpers<color>();
    uint32_t king_jumpers = movers & pos.K;
    uint32_t pawn_jumpers = movers & (~pos.K);


    while (king_jumpers) {
        const uint32_t maske = king_jumpers & ~(king_jumpers - 1u);
        mask_bits<color>(pos, maske);
        add_king_captures<color, KING>(pos, maske, maske, 0, liste);
        mask_bits<color>(pos, maske);
        king_jumpers &= king_jumpers - 1u;
    }

    while (pawn_jumpers) {
        const uint32_t maske = pawn_jumpers & ~(pawn_jumpers - 1u);
        add_king_captures<color, PAWN>(pos, maske, maske, 0, liste);
        pawn_jumpers &= pawn_jumpers - 1u;
    }

}


inline void get_moves(Position &pos, MoveListe &liste) {
    liste.reset();
    if (pos.color == BLACK) {
        loop_captures<BLACK>(pos, liste);
        if (!liste.is_empty())
            return;
        get_silent_moves<BLACK>(pos, liste);
    } else {
        loop_captures<WHITE>(pos, liste);
        if (!liste.is_empty())
            return;
        get_silent_moves<WHITE>(pos, liste);
    }
}

inline void get_captures(Position &pos, MoveListe &liste) {
    liste.reset();
    if (pos.get_color() == BLACK) {
        loop_captures<BLACK>(pos, liste);
    } else {
        loop_captures<WHITE>(pos, liste);
    }
}

#endif //CHECKERSTEST_MGENERATOR_H