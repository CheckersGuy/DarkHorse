//
// Created by robin on 7/30/18.
//

#ifndef CHECKERENGINEX_WEIGHTS_H
#define CHECKERENGINEX_WEIGHTS_H

#include <string>
#include "Position.h"
#include <fstream>
#include <iomanip>
#include "MGenerator.h"
#include "GameLogic.h"
#include <cstring>

constexpr uint32_t region = 13107;
extern uint64_t nodeCounter;

constexpr int powers[] = {1, 5, 25, 125, 625, 3125, 15625, 78125};

constexpr int SIZE = 390625 * 9 * 2;

inline int getIndex(uint32_t region, const Position &pos) {
    //will return the index for a given position
    int index = 0;
    uint32_t pieces = region & (pos.BP | pos.WP);
    int counter = 0;
    while (pieces) {
        uint32_t lsb = (pieces & ~(pieces - 1));
        pieces &= pieces - 1;
        uint32_t current = 0;
        if (((pos.K & pos.BP) & lsb)) {
            current = 1;
        } else if (((pos.K & pos.WP) & lsb)) {
            current = 2;
        } else if (((pos.BP) & lsb)) {
            current = 3;
        } else if (((pos.WP) & lsb)) {
            current = 4;
        }
        index += powers[counter++] * current;
    }
    return index;
}

template<typename T>
class Weights {

public:
    T *weights;

    Weights() {
        weights = new T[SIZE];
        std::memset(weights, 0, sizeof(T) * SIZE);
    }

    ~Weights() {
        delete[] weights;
    }

    int numNonZeroValues() {
        int counter = 0;
        for (int i = 0; i < SIZE; ++i) {
            if (weights[i] != 0) {
                counter++;
            }
        }
        return counter;
    }

    T getMaxValue() {
        T tempValue = -1000;
        for (int i = 0; i < SIZE; ++i) {
            if (weights[i] > tempValue) {
                tempValue = weights[i];
            }
        }
        return tempValue;
    }

    T getMinValue() {
        T tempValue = 1000;
        for (int i = 0; i < SIZE; ++i) {
            if (weights[i] < tempValue) {
                tempValue = weights[i];
            }
        }
        return tempValue;
    }

    void loadWeights(const std::string path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Error: Couldn't find weights, default init" << std::endl;;
        } else {
            stream.read((char *) &weights[0], sizeof(T) * SIZE);
        }
        stream.close();
    }


    void storeWeights(const std::string path) {
        std::ofstream stream(path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Error" << std::endl;
            exit(0);
        }
        std::cout << "Size: " << SIZE << std::endl;
        std::cout << "Test: " << sizeof(T) << std::endl;
        stream.write((char *) &weights[0], sizeof(T) * SIZE);
        stream.close();
    }

    inline Value evaluate(const Position &pos) const {
        int sum = 0;
        int phase = 0;
        const uint32_t nKings = ~pos.K;
        const int rightBlack = __builtin_popcount(pos.BP & nKings & RIGHT_HALF);
        const int rightWhite = __builtin_popcount(pos.WP & nKings & RIGHT_HALF);
        const int leftBlack = __builtin_popcount(pos.BP & nKings & LEFT_HALF);
        const int leftWhite = __builtin_popcount(pos.WP & nKings & LEFT_HALF);

        const int WP = rightWhite + leftWhite;
        const int BP = rightBlack + leftBlack;


        int openingWhite = 0, openingBlack = 0,
                endingWhite = 0, endingBlack = 0,
                opening = 0, ending = 0;

        sum += 100 * (WP - BP);


        phase += WP + BP;
        if (pos.K != 0) {
            const int WK = __builtin_popcount(pos.K & (pos.WP));
            const int BK = __builtin_popcount(pos.K & (pos.BP));
            phase += WK + BK;
            openingWhite += 150 * (WK);
            openingBlack += 150 * BK;
            endingWhite += 110 * (WK);
            endingBlack += 110 * BK;
        }

        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8 * j + i);
                int index = getIndex(curRegion, pos);
                opening += (int) (weights[index + 390625 * (3 * j + i)]);
                ending += (int) (weights[index + 390625 * 9 + 390625 * (3 * j + i)]);
            }
        }


        const int scoreWhite = std::abs(leftWhite - rightWhite);
        const int scoreBlack = std::abs(leftBlack - rightBlack);
        sum += (scoreWhite - scoreBlack) * -2;;

        int phasedScore = 0;
        phasedScore += ((phase * opening + (24 - phase) * ending));
        phasedScore += ((phase * openingWhite + (24 - phase) * endingWhite));
        phasedScore /= 24;
        phasedScore -= ((phase * openingBlack + (24 - phase) * endingBlack)) / 24;

        sum += phasedScore;

        return sum;
    }

    int getSize() {
        return SIZE + 2;
    }

};


//GameWeights is not correct atm

class GameWeights : public Weights<char> {


public:


    void loadWeights(std::string path) {
        Weights<double> myWeights;
        myWeights.loadWeights(path);
        for (int i = 0; i < SIZE; ++i) {
            weights[i] = (char) (myWeights.weights[i]);
        }
    }


    inline Value evaluate(Position pos) const {
        int sum = 0;
        int phase = 0;


        const uint32_t nKings = ~pos.K;
        const int rightBlack = __builtin_popcount(pos.BP & nKings & RIGHT_HALF);
        const int rightWhite = __builtin_popcount(pos.WP & nKings & RIGHT_HALF);
        const int leftBlack = __builtin_popcount(pos.BP & nKings & LEFT_HALF);
        const int leftWhite = __builtin_popcount(pos.WP & nKings & LEFT_HALF);

        const int WP = rightWhite + leftWhite;
        const int BP = rightBlack + leftBlack;


        int openingWhite = 0, openingBlack = 0,
                endingWhite = 0, endingBlack = 0,
                opening = 0, ending = 0;

        sum += 100 * (WP - BP);


        phase += WP + BP;
        if (pos.K != 0) {
            const int WK = __builtin_popcount(pos.K & (pos.WP));
            const int BK = __builtin_popcount(pos.K & (pos.BP));
            phase += WK + BK;
            openingWhite += 150 * (WK);
            openingBlack += 150 * BK;
            endingWhite += 110 * (WK);
            endingBlack += 110 * BK;
        }

        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8 * j + i);
                int index = getIndex(curRegion, pos);
                opening += weights[index + 390625 * (3 * j + i)];
                ending += weights[index + 390625 * 9 + 390625 * (3 * j + i)];
            }
        }

        const int scoreWhite = std::abs(leftWhite - rightWhite);
        const int scoreBlack = std::abs(leftBlack - rightBlack);
        sum += (scoreWhite - scoreBlack) * -2;

        int phasedScore = 0;
        phasedScore += ((phase * opening + (24 - phase) * ending));
        phasedScore += ((phase * openingWhite + (24 - phase) * endingWhite));
        phasedScore /= 24;
        phasedScore -= ((phase * openingBlack + (24 - phase) * endingBlack)) / 24;

        sum += phasedScore;
        return sum;
    }


};


#endif //CHECKERENGINEX_WEIGHTS_H
