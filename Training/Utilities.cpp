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
            if (value.isInRange(lowerBound, upperBound)) {
                hashes.insert(board.getCurrentKey());
                Position currentPos = *board.getPosition();
                data.emplace_back(currentPos);
                std::cout << "Added position" << std::endl;
            }
            return;
        }
        MoveListe liste;
        getMoves(*board.getPosition(), liste);
        for (int i = 0; i < liste.length(); ++i) {
            board.makeMove(liste.liste[i]);
            createNMoveBook(data, N - 1, board, lowerBound, upperBound);
            board.undoMove();
        }
    }

    void createNMoveBook(std::vector<Position> &pos, int N, Board &board) {
        createNMoveBook(pos, N, board, -50, 50);
    }

    Score playGame(Training::TrainingGame &game, Engine &one, Engine &two, Position position, bool print) {
        Board board;
        BoardFactory::setUpPosition(board, position);
        for (int i = 0; i < MAX_MOVE; ++i) {
            if(board.isRepetition()){
                return DRAW;
            }

            MoveListe liste;
            getMoves(*board.getPosition(), liste);
            if (liste.length() == 0) {
                if (board.getMover() == BLACK) {
                    return WHITE_WIN;
                } else if (board.getMover() == WHITE) {
                    return BLACK_WIN;
                }
            }

            game.add(*board.getPosition());

            if (print) {
                board.printBoard();
                std::cout << "\n";
            }


            Value val;
            Move bestMove;

            if (board.getMover() == BLACK) {
                val = one.searchEngine(board, bestMove, MAX_PLY, one.getTimePerMove(), print);
            } else {
                val = two.searchEngine(board, bestMove, MAX_PLY, two.getTimePerMove(), print);
            }
            if (bestMove.isEmpty()) {
                std::cerr << "ERROR" << std::endl;
                return INVALID;
            }
            board.makeMove(bestMove);
        }

        return DRAW;
    }

}