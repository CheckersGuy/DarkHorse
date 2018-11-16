//
// Created by robin on 5/21/18.
//

#ifndef TRAINING_UTILITIES_H
#define TRAINING_UTILITIES_H

#include <fstream>
#include "Board.h"
#include "GameLogic.h"
#include "MGenerator.h"
#include "Training.h"
#include "BoardFactory.h"
#include "Training.h"

#include <unordered_set>

namespace Utilities {


    void createNMoveBook(std::vector<Position> &data, int N, Board &board, Value lowerBound, Value upperBound);

    void createNMoveBook(std::vector<Position>&pos, int N, Board &board);

    void loadPositions(std::vector<Position>& positions,const std::string file);

    void savePositions(std::vector<Position>&positions,const std::string file);

    Score playGame(Engine&engine,Engine& second,Position position,int time, bool print);

    Score playGame(Training::TrainingGame& game,Engine&engine,Engine& second,Position position,int time, bool print);

}


#endif //TRAINING_UTILITIES_H
