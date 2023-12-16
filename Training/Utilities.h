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
#include <iostream>
#include <istream>
#include <iterator>
#include <ostream>
#include <string>
#include <unordered_set>
namespace Utilities {

void fill_hash(std::string path);

void createNMoveBook(std::ofstream &output, int N, Board &board,
                     Value lowerBound, Value upperBound);

} // namespace Utilities

#endif // TRAINING_UTILITIES_H
