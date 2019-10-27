//
// Created by robin on 10.10.19.
//
#include "Move.h"

Move::Move(uint32_t from, uint32_t to) noexcept  : from(from), to(to) {}

Move::Move(uint32_t from, uint32_t to, uint32_t captures) noexcept  : from(from), to(to), captures(captures) {}


bool Move::isCapture() const {
    return captures != 0;
}

bool Move::isPromotion() const {
    return ((to & PROMO_SQUARES_WHITE) != 0) || ((to & PROMO_SQUARES_BLACK) != 0);
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
    return __tzcnt_u32(from);
}

uint32_t Move::getToIndex() const {
    return __tzcnt_u32(to);
}

bool Move::operator==(const Move &other) const {
    return  (to == other.to) && (from == other.from) && (captures == other.captures);
}

bool Move::operator!=(const Move &other) const {
    return !((*this) == other);
}




