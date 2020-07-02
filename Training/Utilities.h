//
// Created by robin on 5/21/18.
//

#ifndef TRAINING_UTILITIES_H
#define TRAINING_UTILITIES_H


#include <filesystem>
#include "Utilities.h"

#include <fstream>
#include "Board.h"
#include "GameLogic.h"
#include "MGenerator.h"
#include <unordered_set>
#include <iterator>
#include <future>
#include <algorithm>

namespace Utilities {
    void createNMoveBook(std::vector<Position> &data, int N, Board &board, Value lowerBound, Value upperBound);

    void createNMoveBook(std::vector<Position> &pos, int N, Board &board);

    void loadPositions(std::vector<Position> &positions, const std::string file);

    void savePositions(std::vector<Position> &positions, const std::string file);

}


#endif //TRAINING_UTILITIES_H
