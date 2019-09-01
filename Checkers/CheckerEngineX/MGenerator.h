//
// Created by Robin on 11.05.2017.
//
#ifndef CHECKERSTEST_MGENERATOR_H
#define CHECKERSTEST_MGENERATOR_H

#include "Board.h"
#include "immintrin.h"
#include "MoveListe.h"
#include "types.h"

template<PieceType type>
inline void addMove(const uint32_t from, const uint32_t to, const uint32_t captures, MoveListe &liste) {
    Move nMove(from, to, captures);
    if constexpr(type == KING) {
        nMove.setPieceType(1);
    }
    liste.addMove(nMove);
}

template<Color color, PieceType type>
inline
void getSilentMoves(const uint32_t nocc, MoveListe &liste, uint32_t movers) {
    while (movers) {
        const uint32_t index = __tzcnt_u32(movers);
        movers &= movers - 1u;
        const uint32_t maske = 1u << index;
        uint32_t squares = defaultShift<color>(maske) | forwardMask<color>(maske);
        squares &= nocc;
        while (squares) {
            const uint32_t next = __tzcnt_u32(squares);
            squares &= squares - 1u;
            addMove<type>(index, next, 0u, liste);
        }
    }
}

template<Color color, PieceType type>
inline
void
addCaptures(const uint32_t orig, const uint32_t current, const uint32_t captures, uint32_t opp,
            const uint32_t movers, const uint32_t pieces, MoveListe &liste) {
    opp ^= captures;
    const uint32_t nocc = ~(opp | pieces);
    const uint32_t temp0 = defaultShift<color>(current) & opp;
    const uint32_t temp1 = forwardMask<color>(current) & opp;
    const uint32_t dest0 = forwardMask<color>(temp0) & nocc;
    const uint32_t dest1 = defaultShift<color>(temp1) & nocc;
    uint32_t temp2 = 0u, temp3 = 0u, dest2 = 0u, dest3 = 0u;
    uint32_t imed = (forwardMask<~color>(dest0) | defaultShift<~color>(dest1));
    imed |= forwardMask<color>(dest2) | defaultShift<color>(dest3);
    uint32_t dest = dest0 | dest1 | dest2 | dest3;
    if (dest == 0u) {
        addMove<type>(orig, __tzcnt_u32(current), captures, liste);
    }
    while (dest) {
        const uint32_t destMask = dest & (-dest);
        const uint32_t capMask = imed & (-imed);
        dest &= dest - 1u;
        imed &= imed - 1u;
        addCaptures<color, type>(orig, destMask, (captures | capMask), opp, movers, pieces, liste);
    }
}

template<Color color, PieceType type>
inline
void loopCaptures(const Position &pos, MoveListe &liste, uint32_t movers) {
    const uint32_t opp = pos.getCurrent<~color>();
    while (movers) {
        const uint32_t index = __tzcnt_u32(movers);
        const uint32_t maske = 1u << index;
        movers ^= maske;
        addCaptures<color, type>(index, maske, 0, opp, movers, (pos.BP | pos.WP), liste);
        movers ^= maske;
        movers &= movers - 1u;
    }
}

template<Color color>
inline void getMovesSide(Position &pos, MoveListe &liste) {
    const uint32_t nocc = ~(pos.BP | pos.WP);
    loopCaptures<color, KING>(pos, liste, pos.getJumpers<color>() & pos.getPieces<KING>());
    loopCaptures<color, PAWN>(pos, liste, pos.getJumpers<color>() & pos.getPieces<PAWN>());
    if (liste.length() > 0)
        return;
    getSilentMoves<color, KING>(nocc, liste, pos.getMovers<color>() & pos.getPieces<KING>());
    getSilentMoves<~color, KING>(nocc, liste, pos.getMovers<color>() & pos.getPieces<KING>());
    getSilentMoves<color, PAWN>(nocc, liste, pos.getMovers<color>() & pos.getPieces<PAWN>());

}

inline void getMoves(Position &pos, MoveListe &liste) {
    if (pos.getColor() == BLACK) {
        getMovesSide<BLACK>(pos, liste);
    } else {
        getMovesSide<WHITE>(pos, liste);
    }
}

inline void getCaptures(Position &pos, MoveListe &liste) {
    if (pos.getColor() == BLACK) {
        loopCaptures<BLACK, KING>(pos, liste, pos.getJumpers<BLACK>() & pos.getPieces<KING>());
        loopCaptures<BLACK, PAWN>(pos, liste, pos.getJumpers<BLACK>() & pos.getPieces<PAWN>());
    } else {
        loopCaptures<WHITE, KING>(pos, liste, pos.getJumpers<WHITE>() & pos.getPieces<KING>());
        loopCaptures<WHITE, PAWN>(pos, liste, pos.getJumpers<WHITE>() & pos.getPieces<PAWN>());
    }
}

#endif //CHECKERSTEST_MGENERATOR_H