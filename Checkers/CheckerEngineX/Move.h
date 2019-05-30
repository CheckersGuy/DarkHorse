//
// Created by Robin on 11.05.2017.
//

#ifndef CHECKERSTEST_MOVE_H
#define CHECKERSTEST_MOVE_H

#include <iostream>
#include <stdint.h>

struct Move {

    uint16_t encoding=0;
    uint32_t captures=0;



    Move(uint16_t from, uint16_t to);

    Move(uint16_t from, uint16_t to, uint32_t captures);

    Move(uint16_t from, uint16_t to, uint16_t pieceType, uint32_t captures);

    Move()=default;

    bool isCapture() const;

    bool isPromotion() const;

    bool isEmpty() const;

    uint16_t getFrom() const;

    uint16_t getTo() const;

    void setFrom(const uint16_t from);

    void setTo(const uint16_t to);

    uint16_t getPieceType() const;

    void setMoveIndex(const uint16_t index);

    uint16_t getMoveIndex() const;

    void setPieceType(const uint16_t type);

    bool operator==(Move other) const;

    bool operator!=(Move other) const;
};

inline bool Move::operator==(Move other) const {
    return (this->captures == other.captures) &&
           (this->getFrom() == other.getFrom() && this->getTo() == other.getTo() &&
            this->getPieceType() == other.getPieceType());
}

inline bool Move::operator!=(Move other) const {
    return !((*this) == other);
}

inline Move::Move(uint16_t from, uint16_t to) {
    this->setFrom(from);
    this->setTo(to);
}

inline Move::Move(uint16_t from, uint16_t to, uint32_t captures) {
    this->setFrom(from);
    this->setTo(to);
    this->captures=captures;
}

inline Move::Move(uint16_t from, uint16_t to, uint16_t pieceType, uint32_t captures) {
    this->setFrom(from);
    this->setTo(to);
    this->setPieceType(pieceType);
    this->captures = captures;

}

inline bool Move::isCapture() const {
    return (this->captures != 0);
}

inline bool Move::isPromotion() const {
    return (getPieceType() == 0 && ((getTo() >> 2) == 0 || (getTo() >> 2) == 7));
}

inline uint16_t Move::getFrom() const {
    return (31 & this->encoding);
}

inline uint16_t Move::getTo() const {
    return (this->encoding & 992) >> 5;
}

inline uint16_t Move::getPieceType() const {
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

inline bool Move::isEmpty() const {
    return (encoding == 0 && captures == 0);
}

inline uint16_t Move::getMoveIndex() const {
    return (this->encoding & (63488)) >> 11;
}


inline void Move::setPieceType(const uint16_t type) {
    this->encoding &= ~1024;
    this->encoding |= type << 10;
}

inline std::ostream &operator<<(std::ostream &stream, const Move move) {
    stream << "From: " << move.getFrom() << " To: " << move.getTo();
    return stream;
}

#endif //CHECKERSTEST_MOVE_H
