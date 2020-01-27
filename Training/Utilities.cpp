//
// Created by robin on 5/21/18.
//

#include "Utilities.h"

namespace Utilities {

    void loadPositions(std::vector<Position> &positions, const std::string file) {
        std::ifstream stream(file);
        std::istream_iterator<Position> begin(stream);
        std::istream_iterator<Position> end;
        std::copy(begin, end, std::back_inserter(positions));
    }

    void savePositions(std::vector<Position> &positions, const std::string file) {
        std::ofstream stream(file);
        std::copy(positions.begin(), positions.end(), std::ostream_iterator<Position>(stream));
    }

    std::unordered_set<uint64_t> hashes;

    void createNMoveBook(std::vector<Position> &data, int N, Board &board, Value lowerBound, Value upperBound) {
        if (hashes.find(board.getCurrentKey()) != hashes.end()) {
            std::cout << "TEST" << std::endl;
            return;
        }
        if (N == 0) {
            Board copy(board);
            Value value = searchValue(copy, MAX_PLY, 70, false);
            if (isInRange(value,lowerBound, upperBound)) {
                hashes.insert(board.getCurrentKey());
                Position currentPos = board.getPosition();
                data.emplace_back(currentPos);
                std::cout << "Added position" << std::endl;
            }
            return;
        }
        MoveListe liste;
        getMoves(board.getPosition(), liste);
        for (int i = 0; i < liste.length(); ++i) {
            board.makeMove(liste[i]);
            createNMoveBook(data, N - 1, board, lowerBound, upperBound);
            board.undoMove();
        }
    }
}