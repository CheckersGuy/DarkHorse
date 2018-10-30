//
// Created by Robin on 11.05.2017.
//

#ifndef CHECKERSTEST_BOARDFACTORY_H
#define CHECKERSTEST_BOARDFACTORY_H

#include "Board.h"
#include <iostream>
#include <fstream>

namespace BoardFactory {
    void setUpStartingPosition(Board &current);

    void setUpPosition(Board &board, Position pos);

    void getBoardFromString(Board &board, std::string);

    void getOpeningPosition(int index, Board &board);
}
#endif //CHECKERSTEST_BOARDFACTORY_H