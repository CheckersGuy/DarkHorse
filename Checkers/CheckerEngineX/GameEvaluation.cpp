//
// Created by Robin on 10.06.2017.
//

#include "GameEvaluation.h"


Value evaluate(Board &board) {
    Position pos = *board.getPosition();
    const Value eval = materialEvaluation(pos)+saveSquares(pos)+balanceScore(pos)+diagonalSquares(pos)+cornerScore(pos);

    if (eval == 0) {
        return board.getMover();
    }
    assert(eval != 0);
    return eval;
}