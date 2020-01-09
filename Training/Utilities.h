//
// Created by robin on 5/21/18.
//

#ifndef TRAINING_UTILITIES_H
#define TRAINING_UTILITIES_H
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/zlib.hpp>
#include <filesystem>

#include <boost/iostreams/copy.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include "Utilities.h"

#include <fstream>
#include "Board.h"
#include "GameLogic.h"
#include "MGenerator.h"
#include "Training.h"
#include "Training.h"
#include <boost/interprocess/streams/bufferstream.hpp>
#include <unordered_set>

namespace Utilities {

    void createNMoveBook(std::vector<Position> &data, int N, Board &board, Value lowerBound, Value upperBound);

    void createNMoveBook(std::vector<Position>&pos, int N, Board &board);

    void loadPositions(std::vector<Position>& positions,const std::string file);

    void savePositions(std::vector<Position>&positions,const std::string file);

    std::vector<std::string> getWhiteSpaceSeperated(const std::string msg);

    std::vector<std::string> getDelimiterSeperated(const std::string message, const std::string delimiter);
}


#endif //TRAINING_UTILITIES_H
