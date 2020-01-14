//
// Created by Robin on 11.05.2017.
//

#ifndef CHECKERSTEST_MOVE_H
#define CHECKERSTEST_MOVE_H


#include <cstdint>
#include "types.h"
#include "Bits.h"
struct Move {
    uint32_t from = 0u;
    uint32_t to = 0u;
    uint32_t captures = 0u;

    bool isCapture() const;

    bool isEmpty() const;

    uint32_t getFrom() const;

    uint32_t getTo() const;

    bool operator==(const Move &other) const;

    bool operator!=(const Move &other) const;

    uint32_t getFromIndex() const;

    uint32_t getToIndex() const;

    bool isPromotion(const uint32_t kings);

};


#endif //CHECKERSTEST_MOVE_H
