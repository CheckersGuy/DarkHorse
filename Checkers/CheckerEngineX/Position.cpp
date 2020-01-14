//
// Created by Robin on 14.01.2018.
//

#include "Position.h"
#include "Zobrist.h"


Position Position::getColorFlip() const {
    Position next;
    next.BP = getMirrored(WP);
    next.WP = getMirrored(BP);
    next.K = getMirrored(K);
    next.color = ~color;
    next.key = key ^ Zobrist::colorBlack;
    return next;
}

Position Position::getStartPosition() {
    Position pos;
    pos.BP = 0u;
    pos.WP = 0u;
    pos.K = 0u;
    for (int i = 0; i <= 11; i++) {
        pos.BP |= 1u << S[i];
    }
    for (int i = 20; i <= 31; i++) {
        pos.WP |= 1u << S[i];
    }
    pos.key = Zobrist::generateKey(pos);
    return pos;
}

bool Position::isEmpty() const {
    return (BP == 0u && WP == 0u);
}

Color Position::getColor() const {
    return color;
}

bool Position::hasJumps(Color col) const {
    if (col == BLACK) {
        return getJumpers<BLACK>() != 0u;
    } else {
        return getJumpers<WHITE>() != 0u;
    }
}

bool Position::hasThreat() const {
    return hasJumps(~getColor());
}

bool Position::isWipe() const {
    return ((getColor() == BLACK && (getMovers<BLACK>() == 0u) && (getJumpers<BLACK>())==0u) || (getColor() == WHITE && (getMovers<WHITE>() == 0u)&& (getJumpers<WHITE>()==0u)));
}

void Position::printPosition() const {
    std::string output;
    uint32_t counter = 32u;
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        if ((row + col) % 2 == 0) {
            std::cout << "[ ]";
        } else {
            if ((row + col + 1) % 2 == 0) {
                counter--;
            }
            uint32_t maske = 1u << (counter);
            if (((BP & K) & maske) == maske) {
                std::cout << "[B]";
            } else if (((BP) & maske) == maske) {
                std::cout << "[0]";
            } else if (((WP & K) & maske) == maske) {
                std::cout << "[W]";
            } else if (((WP) & maske) == maske) {
                std::cout << "[X]";
            } else {
                std::cout << "[ ]";
            }
        }
        if ((i + 1) % 8 == 0) {
            std::cout << "\n";
        }
    }
}

void Position::makeMove(Move &move) {
    assert(!move.isEmpty());
    //setting the piece type
    if (color == BLACK) {
        if (move.isCapture()) {
            WP &= ~move.captures;
            K &= ~move.captures;
        }
        BP &= ~move.from;
        BP |= move.to;

        if (((move.to & PROMO_SQUARES_BLACK) != 0u) && ((move.from & K) == 0))
            K |= move.to;

    } else {
        if (move.isCapture()) {
            BP &= ~move.captures;
            K &= ~move.captures;
        }
        WP &= ~move.from;
        WP |= move.to;

        if (((move.to & PROMO_SQUARES_WHITE) != 0u) && ((move.from & K) == 0))
            K |= move.to;

    }
    if ((move.from & K) != 0) {
        K &= ~move.from;
        K |= move.to;
    }
    this->color = ~this->color;
}

std::istream &operator>>(std::istream &stream, const Position &pos) {
    stream.read((char *) &pos, sizeof(Position));
    return stream;
}

std::ostream &operator<<(std::ostream &stream, const Position &pos) {
    stream.write((char *) &pos, sizeof(Position));
    return stream;
}