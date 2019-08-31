//
// Created by Robin on 11.05.2017.
//
#ifndef CHECKERSTEST_MGENERATOR_H
#define CHECKERSTEST_MGENERATOR_H

#include "Board.h"
#include "immintrin.h"
#include "MoveListe.h"
#include "types.h"

template<Color color>
inline void maskBits(Position &pos, const uint32_t maske) {
    if constexpr (color == BLACK)
        pos.BP ^= maske;
    else
        pos.WP ^= maske;
}

template<PieceType type>
inline void addMove(const uint32_t from, const uint32_t to, const uint32_t captures, MoveListe &liste) {
    Move nMove(from, to, captures);
    if constexpr(type == KING) {
        nMove.setPieceType(1);
    }
    liste.addMove(nMove);
}

template<Color color>
inline
void getSilentMoves(Position &pos, MoveListe &liste) {
    uint32_t movers = pos.getMovers<color>();
    const uint32_t nocc = ~(pos.BP | pos.WP);
    while (movers) {
        const uint32_t index = __tzcnt_u32(movers);
        movers &= movers - 1u;
        const uint32_t maske = 1u << index;
        uint32_t squares = defaultShift<color>(maske) | forwardMask<color>(maske);
        uint16_t pieceType = 0;
        if ((maske & pos.K)) {
            pieceType = 1;
            squares |= forwardMask<~color>(maske) | defaultShift<~color>(maske);
        }
        squares &= nocc;
        while (squares) {
            const uint32_t next = __tzcnt_u32(squares);
            squares &= squares - 1u;
            Move move(index, next);
            move.setPieceType(pieceType);
            liste.addMove(move);
        }
    }
}

template<Color color>
void addPawnCaptures(Position &pos, uint32_t orig, uint32_t current, uint32_t captures, MoveListe &liste) {
    const uint32_t nocc = ~(pos.BP | pos.WP);
    const uint32_t opp = pos.getCurrent<~color>();
    const uint32_t temp0 = defaultShift<color>(current) & opp;
    const uint32_t temp1 = forwardMask<color>(current) & opp;
    uint32_t imed = temp0 | temp1;
    uint32_t dest = (forwardMask<color>(temp0) | defaultShift<color>(temp1)) & nocc;
    imed &= (forwardMask<~color>(dest) | defaultShift<~color>(dest));
    if (dest == 0u) {
        addMove<PAWN>(orig, __tzcnt_u32(current), captures, liste);
    }
    while (dest) {
        uint32_t destMask = dest & (-dest);
        uint32_t capMask = imed & (-imed);
        dest &= dest - 1u;
        imed &= imed - 1u;
        addPawnCaptures<color>(pos, orig, destMask, (captures | capMask), liste);
    }
}

template<Color color>
inline
void
addKingCaptures(Position &pos, const uint32_t orig, const uint32_t current, const uint32_t captures, MoveListe &liste) {
    const uint32_t opp = pos.getCurrent<~color>() ^captures;
    const uint32_t nocc = ~(opp | pos.getCurrent<color>());
    const uint32_t temp0 = defaultShift<color>(current) & opp;
    const uint32_t temp1 = forwardMask<color>(current) & opp;
    const uint32_t temp2 = defaultShift<~color>(current) & opp;
    const uint32_t temp3 = forwardMask<~color>(current) & opp;

    const uint32_t dest0 = forwardMask<color>(temp0) & nocc;
    const uint32_t dest1 = defaultShift<color>(temp1) & nocc;
    const uint32_t dest2 = forwardMask<~color>(temp2) & nocc;
    const uint32_t dest3 = defaultShift<~color>(temp3) & nocc;


    uint32_t imed = (forwardMask<~color>(dest0) | defaultShift<~color>(dest1));
    imed |= forwardMask<color>(dest2) | defaultShift<color>(dest3);
    uint32_t dest = dest0 | dest1 | dest2 | dest3;
    if (dest == 0u) {
        addMove<KING>(orig, __tzcnt_u32(current), captures, liste);
    }
    while (dest) {
        const uint32_t destMask = dest & (-dest);
        const uint32_t capMask = imed & (-imed);
        dest &= dest - 1u;
        imed &= imed - 1u;
        addKingCaptures<color>(pos, orig, destMask, (captures | capMask), liste);
    }
}

template<Color color>
inline
void loopCaptures(Position &pos, MoveListe &liste) {
    uint32_t movers = pos.getJumpers<color>();
    uint32_t pawnMovers = movers & (~pos.K);
    uint32_t kingMovers = movers & pos.K;
    while (kingMovers) {
        const uint32_t index = __tzcnt_u32(kingMovers);
        const uint32_t maske = 1u << index;
        kingMovers &= kingMovers - 1u;
        maskBits<color>(pos, maske);
        addKingCaptures<color>(pos, index, maske, 0, liste);
        maskBits<color>(pos, maske);
    }

    while (pawnMovers) {
        const uint32_t index = __tzcnt_u32(pawnMovers);
        const uint32_t maske = 1u << index;
        pawnMovers &= pawnMovers - 1u;
        addPawnCaptures<color>(pos, index, maske, 0, liste);
    }

}

inline void getMoves(Position &pos, MoveListe &liste) {
    if (pos.color == BLACK) {
        loopCaptures<BLACK>(pos, liste);
        if (liste.moveCounter > 0)
            return;
        getSilentMoves<BLACK>(pos, liste);
    } else {
        loopCaptures<WHITE>(pos, liste);
        if (liste.moveCounter > 0)
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