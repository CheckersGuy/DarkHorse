//
// Created by Robin on 14.01.2018.
//

#include "Position.h"


Position Position::getColorFlip() {
    Position pos;
    pos.color =~color;

    for (int index = 0; index < 16; ++index) {
        int col = index % 4;
        int row = index / 4;
        int newIndex = 3 - col + (7 - row) * 4;
        //BP
        uint32_t upValue = (BP & (1 << index)) >> index;
        uint32_t downValue = (BP & (1 << newIndex)) >> newIndex;
        pos.BP |= (1 << index) * downValue;
        pos.BP |= (1 << newIndex) * upValue;

        upValue = (WP & (1 << index)) >> index;
        downValue = (WP & (1 << newIndex)) >> newIndex;
        pos.WP |= (1 << index) * downValue;
        pos.WP |= (1 << newIndex) * upValue;

        upValue = (K & (1 << index)) >> index;
        downValue = (K & (1 << newIndex)) >> newIndex;
        pos.K |= (1 << index) * downValue;
        pos.K |= (1 << newIndex) * upValue;
    }
    //finally swapping the pieces
     uint32_t temp =pos.BP;
     pos.BP=pos.WP;
     pos.WP=temp;

    return pos;
}

bool Position::isEmpty() {
    return (BP == 0 && WP == 0 && K == 0);
}

Color Position::getColor() {
    return color;
}

bool Position::hasJumps(Color color) {
    if (color == BLACK) {
        return getJumpersBlack() != 0;
    } else if (color == WHITE) {
        return getJumpersWhite() != 0;
    }
}

bool Position::hasThreat() {
    //if the other play can jump
    return hasJumps(~getColor());
}

bool Position::isLoss() {
    return ((getJumpersWhite() == 0 && getMoversWhite() == 0) || (getJumpersBlack() == 0 && getMoversBlack() == 0));
}

bool Position::isWipe() {
    return ((getColor() == BLACK && getMoversBlack() == 0) || (getColor() == WHITE && getMoversWhite() == 0));
}

void Position::printPosition() {
    std::string output;
    int counter = 32;
    for (int i = 0; i < 64; i++) {
        int row = i / 8;
        int col = i % 8;
        if ((row + col) % 2 == 0) {
            std::cout << "[ ]";
        } else {
            if ((row + col + 1) % 2 == 0) {
                counter--;
            }
            int maske = 1 << (counter);
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


void Position::makeMove(Move move) {
    assert(!move.isEmpty());
    //setting the piece type
    if (color == BLACK) {
        if (move.isCapture()) {
            WP &= ~move.captures;
            K &= ~move.captures;
        }
        BP &= ~(1 << move.getFrom());
        BP |= 1 << move.getTo();

        if (move.getPieceType() == 0 && move.getTo() >> 2 == 7)
            K |= 1 << move.getTo();

    } else {
        if (move.isCapture()) {
            BP &= ~move.captures;
            K &= ~move.captures;
        }
        WP &= ~(1 << move.getFrom());
        WP |= 1 << move.getTo();

        if (move.getPieceType() == 0 && move.getTo() >> 2 == 0)
            K |= 1 << move.getTo();

    }
    if (move.getPieceType() == 1) {
        K &= ~((1 << move.getFrom()));
        K |= (1 << move.getTo());
    }
    this->color = ~this->color;
}

void Position::undoMove(Move move) {
    this->color = ~this->color;
    assert(!move.isEmpty());
    if (color == BLACK) {
        BP &= ~(1 << move.getTo());
        BP |= (1 << move.getFrom());

        if (move.isCapture()) {
            WP |= move.captures;
        }

        if (move.getPieceType() == 0 && (move.getTo() >> 2) == 7)
            K &= ~(1 << move.getTo());

    } else {
        WP &= ~(1 << move.getTo());
        WP |= (1 << move.getFrom());
        if (move.isCapture()) {
            BP |= move.captures;
        }
        if (move.getPieceType() == 0 && (move.getTo() >> 2) == 0)
            K &= ~(1 << move.getTo());
    }

    if (move.getPieceType() == 1) {
        K |= ((1 << move.getFrom()));
        K &= ~(1 << move.getTo());
    }

}


uint32_t Position::getMoversWhite() {
    const uint32_t nocc = ~(BP | WP);
    const uint32_t WK = (WP & K);
    uint32_t movers = (nocc << 4) & WP;
    movers |= ((nocc & MASK_L3) << 3) & WP;
    movers |= ((nocc & MASK_L5) << 5) & WP;
    if (WK) {
        movers |= (nocc >> 4) & WK;
        movers |= ((nocc & MASK_R3) >> 3) & WK;
        movers |= ((nocc & MASK_R5) >> 5) & WK;
    }
    return movers;
}


uint32_t Position::getMoversBlack() {
    const uint32_t nocc = ~(BP | WP);
    const uint32_t BK = BP & K;
    uint32_t movers = (nocc >> 4) & BP;
    movers |= ((nocc & MASK_R3) >> 3) & BP;
    movers |= ((nocc & MASK_R5) >> 5) & BP;
    if (BK != 0) {
        movers |= (nocc << 4) & BK;
        movers |= ((nocc & MASK_L3) << 3) & BK;
        movers |= ((nocc & MASK_L5) << 5) & BK;
    }
    return movers;
}


uint32_t Position::getJumpersBlack() {
    const uint32_t nocc = ~(BP | WP);

    const uint32_t BK = (BP & K);
    uint32_t movers = 0;
    uint32_t temp = (nocc >> 4) & WP;
    if (temp != 0) {
        movers |= (((temp & MASK_R3) >> 3) | ((temp & MASK_R5) >> 5)) & BP;
    }
    temp = (((nocc & MASK_R3) >> 3) | ((nocc & MASK_R5) >> 5)) & WP;
    if (temp != 0) {
        movers |= ((temp >> 4)) & BP;
    }

    if (BK != 0) {
        temp = (nocc << 4) & WP;
        if (temp != 0) {
            movers |= (((temp & MASK_L3) << 3) | ((temp & MASK_L5) << 5)) & BK;
        }
        temp = (((nocc & MASK_L3) << 3) | ((nocc & MASK_L5) << 5)) & WP;

        if (temp != 0) {
            movers |= ((temp << 4)) & BK;
        }
    }
    return movers;
}

uint32_t Position::getJumpersWhite() {
    const uint32_t nocc = ~(BP | WP);

    const uint32_t WK = WP & K;
    uint32_t movers = 0;
    uint32_t temp = (nocc << 4) & BP;
    if (temp != 0) {
        movers |= (((temp & MASK_L3) << 3) | ((temp & MASK_L5) << 5)) & WP;
    }
    temp = (((nocc & MASK_L3) << 3) | ((nocc & MASK_L5) << 5)) & BP;
    if (temp != 0) {
        movers |= ((temp << 4)) & WP;
    }
    if (WK != 0) {
        temp = (nocc >> 4) & BP;
        if (temp != 0) {
            movers |= (((temp & MASK_R3) >> 3) | ((temp & MASK_R5) >> 5)) & WK;
        }
        temp = (((nocc & MASK_R3) >> 3) | ((nocc & MASK_R5) >> 5)) & BP;
        if (temp != 0) {
            movers |= ((temp >> 4)) & WK;
        }
    }
    return movers;
}

