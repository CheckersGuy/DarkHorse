//
// Created by Robin on 11.05.2017.
//
#ifndef CHECKERSTEST_MGENERATOR_H
#define CHECKERSTEST_MGENERATOR_H

#include "Board.h"
#include "MoveListe.h"


enum MoveType {
    PawnMove, KingMove, PromoMove
};


struct PerftCallBack {
    size_t num_nodes{0ull};

    inline void operator()(uint32_t &maske, uint32_t &next) {
        this->num_nodes++;
    };

    inline void operator()(uint32_t &from, uint32_t &to, uint32_t &captures) {
        this->num_nodes++;
    };

    void visit(uint32_t &maske, uint32_t &next);

    void visit(uint32_t &from, uint32_t &to, uint32_t &captures);


};

//seperate struct for other things like, Pawnmove,kingmove,kingjump,pawnjump,promotion-move


template<Color color>
inline void mask_bits(Position &pos, const uint32_t maske) {
    if constexpr (color == BLACK)
        pos.BP ^= maske;
    else
        pos.WP ^= maske;
}

template<Color color, typename CallBack>
inline
void get_silent_moves(const Position &pos, CallBack &&call_back) {
    uint32_t pawn_movers = pos.get_movers<color>() & (~pos.K);
    uint32_t king_movers = pos.get_movers<color>() & pos.K;
    const uint32_t nocc = ~(pos.BP | pos.WP);

    while (pawn_movers) {
        uint32_t maske = pawn_movers & ~(pawn_movers - 1u);
        uint32_t squares = defaultShift<color>(maske) | forwardMask<color>(maske);
        squares &= nocc;
        while (squares) {
            uint32_t next = squares & ~(squares - 1u);
            std::forward<CallBack>(call_back)(maske, next);
            squares &= squares - 1u;
        }
        pawn_movers &= pawn_movers - 1u;
    }
    while (king_movers) {
        uint32_t maske = king_movers & ~(king_movers - 1u);
        uint32_t squares = get_neighbour_squares<color>(maske);
        squares &= nocc;
        while (squares) {
            uint32_t next = squares & ~(squares - 1u);
            std::forward<CallBack>(call_back)(maske, next);
            squares &= squares - 1u;
        }
        king_movers &= king_movers - 1u;
    }

}

template<Color color, PieceType type, typename CallBack>
inline
void
add_king_captures(const Position &pos, uint32_t orig, uint32_t current, uint32_t captures,
                  CallBack &&call_back) {
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
        call_back(orig, current, captures);
    }
    while (dest) {
        uint32_t destMask = dest & ~(dest - 1u);
        uint32_t capMask = imed & ~(imed - 1u);
        dest &= dest - 1u;
        imed &= imed - 1u;
        add_king_captures<color, type>
                (pos, orig, destMask, (captures | capMask), std::forward<CallBack>(call_back));
    }
}

template<Color color, typename CallBack>
inline
void loop_captures(Position &pos, CallBack &&call_back) {
    uint32_t movers = pos.get_jumpers<color>();
    uint32_t king_jumpers = movers & pos.K;
    uint32_t pawn_jumpers = movers & (~pos.K);

    while (king_jumpers) {
        const uint32_t maske = king_jumpers & ~(king_jumpers - 1u);
        mask_bits<color>(pos, maske);
        add_king_captures<color, KING>(pos, maske, maske, 0, std::forward<CallBack>(call_back));
        mask_bits<color>(pos, maske);
        king_jumpers &= king_jumpers - 1u;
    }

    while (pawn_jumpers) {
        const uint32_t maske = pawn_jumpers & ~(pawn_jumpers - 1u);
        add_king_captures<color, PAWN>(pos, maske, maske, 0, std::forward<CallBack>(call_back));
        pawn_jumpers &= pawn_jumpers - 1u;
    }

}


template<typename Accumulator>
inline void get_moves(Position &pos, Accumulator &accu) {
    using accu_type = std::decay_t<Accumulator>;


    if constexpr(std::is_same_v<accu_type, MoveListe>) {
        accu.reset();
        if (pos.color == BLACK) {
            loop_captures<BLACK>(pos, accu);
            if (!accu.is_empty())
                return;
            get_silent_moves<BLACK>(pos, accu);
        } else {
            loop_captures<WHITE>(pos, accu);
            if (!accu.is_empty())
                return;
            get_silent_moves<WHITE>(pos, accu);
        }
    } else {
        if (pos.color == BLACK) {
            if (pos.has_jumps<BLACK>())
                loop_captures<BLACK>(pos, accu);
            else
                get_silent_moves<BLACK>(pos, accu);

        } else {
            if (pos.has_jumps<WHITE>())
                loop_captures<WHITE>(pos, accu);
            else
                get_silent_moves<WHITE>(pos, accu);
        }
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