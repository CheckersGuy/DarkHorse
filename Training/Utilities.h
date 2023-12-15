//
// Created by robin on 5/21/18.
//

#ifndef TRAINING_UTILITIES_H
#define TRAINING_UTILITIES_H
#include "Board.h"
#include "GameLogic.h"
#include "MGenerator.h"
#include <algorithm>
#include <fstream>
#include <istream>
#include <iterator>
#include <ostream>
#include <unordered_set>
namespace Utilities {

void createNMoveBook(std::ofstream &output, int N, Board &board,
                     Value lowerBound, Value upperBound);

} // namespace Utilities

#endif // TRAINING_UTILITIES_H
