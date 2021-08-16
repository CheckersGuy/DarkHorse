//
// Created by robin on 5/21/18.
//

#ifndef TRAINING_UTILITIES_H
#define TRAINING_UTILITIES_H


#include <fstream>
#include "Board.h"
#include "GameLogic.h"
#include "MGenerator.h"
#include <unordered_set>
#include <ostream>
#include <istream>
#include <iterator>

namespace Utilities {
    extern std::unordered_set<uint64_t> hashes;

    template<typename T, typename Iterator>
    void write_to_binary(Iterator begin, Iterator end, std::string output, std::ios::openmode mode = std::ios::out | std::ios::binary) {
        std::ofstream stream(output, mode);
        std::copy(begin, end, std::ostream_iterator<T>(stream));
    }


    template<typename T, typename OutIter>
    void read_binary(OutIter out, std::string input) {
        std::ifstream stream(input, std::ios::binary);
        std::istream_iterator<T> begin(stream);
        std::istream_iterator<T> end{};
        std::copy(begin, end, out);
    }

    template<typename OutIter>
    void createNMoveBook(OutIter iter, int N, Board &board, Value lowerBound, Value upperBound) {
        if (hashes.find(board.getCurrentKey()) != hashes.end()) {
            return;
        }
        if (N == 0) {
            Board copy(board);
            Move best;
            Value value = searchValue(board, best, MAX_PLY, 40, false);
            if (value >= lowerBound && value <= upperBound) {
                hashes.insert(board.getCurrentKey());
                Position currentPos = board.getPosition();
                *iter = currentPos;
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
