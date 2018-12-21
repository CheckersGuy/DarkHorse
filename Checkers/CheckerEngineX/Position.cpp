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
        return getJumpers<BLACK>() != 0;
    } else if (color == WHITE) {
        return getJumpers<WHITE>() != 0;
    }
}

bool Position::hasThreat() {
    //if the other play can jump
    return hasJumps(~getColor());
}

bool Position::isLoss() {
    return ((getJumpers<WHITE>() == 0 && getMovers<WHITE>() == 0) || (getJumpers<WHITE>() == 0 && getMovers<WHITE>() == 0));
}

bool Position::isWipe() {
    return ((getColor() == BLACK && getMovers<BLACK>() == 0) || (getColor() == WHITE && getMovers<WHITE>() == 0));
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
