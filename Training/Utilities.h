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
#include<proto/Training.pb.h>
#include <ostream>
#include <istream>

namespace Utilities {
    extern std::unordered_set<uint64_t> hashes;

    template<typename T, typename Iterator>
    void write_to_binary(Iterator begin, Iterator end, std::string output) {
        std::ofstream stream(output, std::ios::binary);
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
            Value value = searchValue(board, best, MAX_PLY, 20, false);
            if (value >= lowerBound && value <= upperBound) {
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



   /*template<typename Iterator>
    void to_binary_data(Iterator begin, Iterator end, std::string output) {
        std::ofstream stream(output, std::ios::binary | std::ios::app);
        if (!stream.good()) {
            throw std::ios::failure("Could not open the stream");
        }
        auto lambda = [&stream](const Training::Position &pos) {
            int color = (pos.mover() == Training::BLACK) ? 0 : 1;
            uint32_t WP =pos.wp();
            uint32_t BP =pos.bp();
            uint32_t K =pos.k();

            int result;
            if (pos.result() == Training::BLACK_WON)
                result = -1;
            else if (pos.result() == Training::WHITE_WON)
                result = 1;
            else
                result = 0;
            stream.write((char *) &WP, sizeof(uint32_t));
            stream.write((char *) &BP, sizeof(uint32_t));
            stream.write((char *) &K, sizeof(uint32_t));
            stream.write((char *) &color, sizeof(int));
            stream.write((char *) &result, sizeof(int));
        };

    }*/
}


#endif //TRAINING_UTILITIES_H
