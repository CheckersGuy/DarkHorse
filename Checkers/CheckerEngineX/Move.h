//
// Created by Robin on 11.05.2017.
//

#ifndef CHECKERSTEST_MOVE_H
#define CHECKERSTEST_MOVE_H

#include <iostream>
#include <stdint.h>


extern "C" void testing();


class Move {

public:
    uint16_t encoding = 0;
    uint32_t captures = 0;


    Move(uint16_t from, uint16_t to);

    Move(uint16_t from, uint16_t to, uint16_t pieceType);

    Move(uint16_t from, uint16_t to, uint16_t pieceType, uint32_t captures);

    Move() = default;

    bool isCapture();

    bool isPromotion();

    bool isEmpty();

    uint16_t getFrom();

    uint16_t getTo();

    void setFrom(const uint16_t from);

    void setTo(const uint16_t to);

    uint8_t getPieceType();

    void setMoveIndex(const uint16_t index);

    uint16_t getMoveIndex();

    void setPieceType(const uint16_t type);

    bool operator==(Move other);

    bool operator!=(Move other);
};

inline bool Move::operator==(Move other) {
    return (this->captures == other.captures) && (this->getFrom() == other.getFrom() && this->getTo() == other.getTo()&& this->getPieceType() == other.getPieceType());
}

inline bool Move::operator!=(Move other) {
    return !((*this) == other);
}

inline Move::Move(uint16_t from, uint16_t to) {
    this->setFrom(from);
    this->setTo(to);
}

inline Move::Move(uint16_t from, uint16_t to, uint16_t pieceType) {
    this->setFrom(from);
    this->setTo(to);
    this->setPieceType(pieceType);
}

inline Move::Move(uint16_t from, uint16_t to, uint16_t pieceType, uint32_t captures) {
    this->setFrom(from);
    this->setTo(to);
    this->setPieceType(pieceType);
    this->captures = captures;

}

inline bool Move::isCapture() {
    return (this->captures != 0);
}

inline bool Move::isPromotion() {
    return (getPieceType() == 0 && ((getTo() >> 2) == 0 || (getTo() >> 2) == 7));
}

inline uint16_t Move::getFrom() {
    return (31 & this->encoding);
}

inline uint16_t Move::getTo() {
    return (this->encoding & 992) >> 5;
}

inline uint8_t Move::getPieceType() {
    return (1024 & this->encoding) >> 10;
}

inline void Move::setFrom(const uint16_t from) {
    this->encoding &= ~(31);
    this->encoding |= from;
}

inline void Move::setTo(const uint16_t to) {
    this->encoding &= ~992;
    this->encoding |= to << 5;
}

inline void Move::setMoveIndex(uint16_t index) {
    this->encoding &= ~63488;
    this->encoding |= index << 11;
}

inline bool Move::isEmpty() {
    return (encoding == 0 && captures == 0);
}

inline uint16_t Move::getMoveIndex() {
    return (this->encoding & (63488)) >> 11;
}


inline void Move::setPieceType(const uint16_t type) {
    this->encoding &= ~1024;
    this->encoding |= type << 10;
}

inline std::ostream &operator<<(std::ostream &stream, Move move) {
    stream << "From: " << move.getFrom() << " To: " << move.getTo();
    return stream;
}

#endif //CHECKERSTEST_MOVE_H
