//
// Created by robin on 10.10.19.
//
#include "Move.h"

bool Move::isCapture() const {
    return captures != 0u;
}

bool Move::isEmpty() const {
    return (from == 0u) && (to == 0u) && (captures == 0u);
}

uint32_t Move::getFromIndex() const {
    return Bits::bitscan_foward(from);
}

uint32_t Move::getToIndex() const {
    return Bits::bitscan_foward(to);
}

bool Move::operator==(const Move &other) const {
    return (to == other.to) && (from == other.from) && (captures == other.captures);
}

bool Move::operator!=(const Move &other) const {
    return !((*this) == other);
}

bool Move::isPromotion(const uint32_t kings) {
    constexpr uint32_t promo_squares = PROMO_SQUARES_BLACK | PROMO_SQUARES_WHITE;
    return (((from & kings) == 0u) && ((to & promo_squares) != 0u));
}


