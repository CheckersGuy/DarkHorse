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
    extern std::unordered_set<uint64_t> hashes;

    template<typename OutIter>
    void createNMoveBook(OutIter iter, int N, Board &board, Value lowerBound, Value upperBound) {
        if (hashes.find(board.getCurrentKey()) != hashes.end()) {
            return;
        }
        if (N == 0) {
            Board copy(board);
            Move best;
            Value value = searchValue(copy, lowerBound, upperBound, best, MAX_PLY, 100, false);
            if (value>=lowerBound && value<=upperBound) {
                hashes.insert(board.getCurrentKey());
                Position currentPos = board.getPosition();
                *iter = currentPos;
                iter++;
                std::cout << "Added position" << std::endl;
            }
            return;
        }
        MoveListe liste;
        getMoves(board.getPosition(), liste);
        for (int i = 0; i < liste.length(); ++i) {
            board.makeMove(liste[i]);
            createNMoveBook(iter, N - 1, board, lowerBound, upperBound);
            board.undoMove();
        }
    }
}


#endif //TRAINING_UTILITIES_H
