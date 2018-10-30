//
// Created by robin on 5/21/18.
//

#include "Utilities.h"

namespace Utilities {


    void loadPositions(std::vector<Position> &positions, const std::string file) {
        std::ifstream stream;
        stream.open(file);
        if (!stream.good())
            return exit(EXIT_FAILURE);

        while (!stream.eof()) {
            Position current;
            stream.read((char *) &current, sizeof(Position));
            if (current.isEmpty())
                continue;
            positions.push_back(current);
        }
        stream.close();

    }

    void savePositions(std::vector<Position> &positions, const std::string file) {
        std::ofstream stream;
        stream.open(file);
        if (!stream.good())
            exit(0);

        for (Position pos : positions) {
            if (pos.isEmpty())
                continue;
            stream.write((char *) &pos, sizeof(Position));
        }
        stream.close();
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
        getMoves(board, liste);
        for (int i = 0; i < liste.length(); ++i) {
            board.makeMove(liste.liste[i]);
            createNMoveBook(data, N - 1, board, lowerBound, upperBound);
            board.undoMove();
        }
    }

    void createNMoveBook(std::vector<Position> &pos, int N, Board &board) {
        createNMoveBook(pos, N, board, -50, 50);
    }


    Score playGame(Engine &one, Engine &two, Position position, int time, bool print) {
        Board board;
        BoardFactory::setUpPosition(board, position);
        for (int i = 0; i < MAX_MOVE; ++i) {
            if (board.isRepetition())
                return DRAW;

            MoveListe liste;
            getMoves(board, liste);
            if (liste.length() == 0) {
                if (board.getMover() == BLACK) {
                    return WHITE_WIN;
                } else if (board.getMover() == WHITE) {
                    return BLACK_WIN;
                }
            }

            if (print) {
                board.printBoard();
                std::cout << "\n";
            }
            Value val;
            Move bestMove;


            if (board.getMover() == BLACK) {
                val = one.searchEngine(board, bestMove, MAX_PLY, time, print);
            } else {
                val = two.searchEngine(board, bestMove, MAX_PLY, time, print);
            }
            if (bestMove.isEmpty()) {
                std::cerr << "ERROR-Empty" << std::endl;
                return INVALID;
            }
        }

        return DRAW;
    }

    Score playGame(Training::TrainingGame &game, Engine &one, Engine &two, Position position, int time, bool print) {
        Board board;
        BoardFactory::setUpPosition(board, position);
        std::cerr << "GameStart" << std::endl;
        for (int i = 0; i < MAX_MOVE; ++i) {
            if (board.isRepetition())
                return DRAW;
            MoveListe liste;
            getMoves(board, liste);
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

                val = one.searchEngine(board, bestMove, MAX_PLY, time, print);
            } else {
                val = two.searchEngine(board, bestMove, MAX_PLY, time, print);
            }
            if (bestMove.isEmpty()) {
                std::cerr << "ERROR" << std::endl;
                return INVALID;
            }
        }

        return DRAW;
    }

}