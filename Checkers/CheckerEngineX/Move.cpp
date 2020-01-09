//
// Created by robin on 10.10.19.
//
#include "Move.h"

bool Move::isCapture() const {
    return captures != 0;
}


uint32_t Move::getFrom() const {
    return from;
}

uint32_t Move::getTo() const {
    return to;
}

bool Move::isEmpty() const {
    return (from == 0) && (to == 0) && (captures == 0);
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
    //current is the position before the move is made
    return (((to & PROMO_SQUARES_WHITE) != 0) || ((to & PROMO_SQUARES_BLACK) != 0)) && ((from & kings) == 0u);
}




