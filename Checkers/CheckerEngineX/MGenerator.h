//
// Created by Robin on 11.05.2017.
//
#ifndef CHECKERSTEST_MGENERATOR_H
#define CHECKERSTEST_MGENERATOR_H

#include "Board.h"
#include "MoveListe.h"


template<Color color>
inline void maskBits(Position &pos, const uint32_t maske) {
    if constexpr (color == BLACK)
        pos.BP ^= maske;
    else
        pos.WP ^= maske;
}


template<Color color>
inline
void getSilentMoves(const Position &pos, MoveListe &liste) {
    uint32_t movers = pos.getMovers<color>();
    const uint32_t nocc = ~(pos.BP | pos.WP);
    while (movers) {
        const uint32_t maske = movers & ~(movers - 1u);
        uint32_t squares = defaultShift<color>(maske) | forwardMask<color>(maske);
        squares |= forwardMask<~color>(maske & pos.K) | defaultShift<~color>(maske & pos.K);
        squares &= nocc;
        while (squares) {
            const uint32_t next = squares & ~(squares - 1u);
            Move move{maske, next};
            liste.addMove(move);
            squares &= squares - 1u;
        }
        movers &= movers - 1u;
    }
}

template<Color color, PieceType type>
inline
void
addKingCaptures(const Position &pos, const uint32_t orig, const uint32_t current, const uint32_t captures,
                MoveListe &liste) {
    const uint32_t opp = pos.getCurrent<~color>() ^captures;
    const uint32_t nocc = ~(opp | pos.getCurrent<color>());
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
        liste.addMove(Move{orig, current, captures});
    }
    while (dest) {
        uint32_t destMask = dest & ~(dest - 1u);
        uint32_t capMask = imed & ~(imed - 1u);
        dest &= dest - 1u;
        imed &= imed - 1u;
        addKingCaptures<color, type>(pos, orig, destMask, (captures | capMask), liste);
    }
}

template<Color color>
inline
void loopCaptures(Position &pos, MoveListe &liste) {
    uint32_t movers = pos.getJumpers<color>();
    uint32_t kingJumpers = movers & pos.K;
    uint32_t pawnJumpers = movers & (~pos.K);
    while (kingJumpers) {
        const uint32_t maske = kingJumpers & ~(kingJumpers - 1u);
        maskBits<color>(pos, maske);
        addKingCaptures<color, KING>(pos, maske, maske, 0, liste);
        maskBits<color>(pos, maske);
        kingJumpers &= kingJumpers - 1u;
    }
    while (pawnJumpers) {
        const uint32_t maske = pawnJumpers & ~(pawnJumpers - 1u);
        addKingCaptures<color, PAWN>(pos, maske, maske, 0, liste);
        pawnJumpers &= pawnJumpers - 1u;
    }
}


inline void getMoves(Position &pos, MoveListe &liste) {
    if (pos.color == BLACK) {
        loopCaptures<BLACK>(pos, liste);
        if (!liste.isEmpty())
            return;
        getSilentMoves<BLACK>(pos, liste);
    } else {
        loopCaptures<WHITE>(pos, liste);
        if (!liste.isEmpty())
            return;
        getSilentMoves<WHITE>(pos, liste);
    }
}

inline void getCaptures(Position &pos, MoveListe &liste) {
    if (pos.getColor() == BLACK) {
        loopCaptures<BLACK>(pos, liste);
    } else {
        loopCaptures<WHITE>(pos, liste);
    }
}

#endif //CHECKERSTEST_MGENERATOR_H