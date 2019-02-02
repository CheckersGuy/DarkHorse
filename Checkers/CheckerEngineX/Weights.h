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
#include "FixPoint.h"

constexpr uint32_t region = 13107;
constexpr size_t powers[] = {1, 5, 25, 125, 625, 3125, 15625, 78125};

constexpr size_t SIZE = 390625 * 9 * 2;
using Eval =FixPoint<short,4>;

inline size_t getIndex(uint32_t region, const Position &pos) {
    //will return the index for a given position
    size_t index = 0;
    uint32_t pieces = region & (pos.BP | pos.WP);
    size_t counter = 0;
    while (pieces) {
        uint32_t lsb = (pieces & ~(pieces - 1));
        pieces &= pieces - 1;
        uint32_t current = 0;
        if (((pos.BP & (~pos.K)) & lsb)) {
            current = 3;
        } else if (((pos.WP & (~pos.K)) & lsb)) {
            current = 4;
        } else if (((pos.K & pos.BP) & lsb)) {
            current = 1;
        } else if (((pos.K & pos.WP) & lsb)) {
            current = 2;
        }
        index += powers[counter++] * current;
    }
    return index;
}

template<typename T>
class Weights {

public:
    std::unique_ptr<T[]> weights;

    Weights() {
        this->weights = std::make_unique<T[]>(SIZE);
        std::memset(weights.get(), 0, sizeof(T) * SIZE);
    }

    Weights(const Weights<T> &other) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::memcpy(this->weights.get(), other.weights.get(), sizeof(T) * SIZE);
    }

    Weights(Weights &&other) {
        this->weights = std::move(other.weights);
        other.weights = nullptr;
    }

    size_t numNonZeroValues() {
        return std::count_if(weights.get(), weights.get() + SIZE, [](T val) { return static_cast<int>(val) != 0; });
    }

    T getNorm() {
        T current = 0;
        for (size_t i = 0; i < SIZE; ++i) {
            current += weights[i] * weights[i];
        }
        current = std::sqrt(current);
        return current;
    }

    T getMaxValue() {
        return *std::max_element(weights.get(), weights.get() + SIZE);
    }

    T getMinValue() {
        return *std::min_element(weights.get(), weights.get() + SIZE);
    }

    void loadWeights(const std::string path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Error: Couldn't find weights, default init" << std::endl;;
        } else {
            size_t counter = 0;
            while (!stream.eof()) {
                uint32_t runLength = 0;
                stream.read((char *) &runLength, sizeof(uint32_t));
                double value;
                stream.read((char *) &value, sizeof(double));
                for (size_t i = 0; i < runLength; ++i) {
                    weights[counter] = value;
                    counter++;
                }
            }
        }
        stream.close();
    }


    void storeWeights(const std::string path) {
        std::ofstream stream(path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Error couldnt store weights" << std::endl;
            exit(0);
        }
        for (size_t i = 0; i < SIZE; ++i) {
            uint32_t runLength = 1;
            double value = weights[i];
            while (i < SIZE && weights[i] == weights[i + 1]) {
                ++i;
                ++runLength;
            }
            stream.write((char *) &runLength, sizeof(uint32_t));
            stream.write((char *) &value, sizeof(double));
        }
        stream.close();
    }

    inline Value evaluate(Position pos) const {
        const Color color = pos.getColor();
        const uint32_t nKings = ~pos.K;
        const int rightBlack = __builtin_popcount(pos.BP & nKings & RIGHT_HALF);
        const int rightWhite = __builtin_popcount(pos.WP & nKings & RIGHT_HALF);
        const int leftBlack = __builtin_popcount(pos.BP & nKings & LEFT_HALF);
        const int leftWhite = __builtin_popcount(pos.WP & nKings & LEFT_HALF);

        const int WP = rightWhite + leftWhite;
        const int BP = rightBlack + leftBlack;


        int openingWhite = 0, openingBlack = 0,
                endingWhite = 0, endingBlack = 0;

        Eval opening=0,ending=0;

        int sum = 100 * (WP - BP);


        int phase = WP + BP;
        if (pos.K != 0) {
            const int WK = __builtin_popcount(pos.K & (pos.WP));
            const int BK = __builtin_popcount(pos.K & (pos.BP));
            phase += WK + BK;
            openingWhite += 150 * (WK);
            openingBlack += 150 * BK;
            endingWhite += 110 * (WK);
            endingBlack += 110 * BK;

        }

        if (pos.getColor() == BLACK) {
            pos = pos.getColorFlip();
        }
        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8 * j + i);
                size_t indexOpening = 18 * getIndex(curRegion, pos) + 3 * j + i;
                size_t indexEnding = 18 * getIndex(curRegion, pos) + 9 + 3 * j + i;
                opening +=   weights[indexOpening]*static_cast<int>(color);
                ending +=  weights[indexEnding]*static_cast<int>(color);
            }
        }

        Eval phasedPatterns;
        Eval phaseTemp =phase;
        Eval opFactor =phase;
        Eval endFactor=24-phase;
        opFactor/=24;
        endFactor/=24;

        phasedPatterns =opFactor*opening+endFactor*ending;

        int phasedScore = 0;

        phasedScore += ((phase * openingWhite + (24 - phase) * endingWhite));
        phasedScore /= 24;
        phasedScore -= ((phase * openingBlack + (24 - phase) * endingBlack)) / 24;
        sum += phasedScore;

        int balanceWhite = std::abs(leftWhite - rightWhite);
        int balanceBlack = std::abs(leftBlack - rightBlack);
        sum += 2 * (balanceBlack - balanceWhite);
        sum+=phasedPatterns.round();


        return sum;
    }

    T &operator[](size_t index) {
        return weights[index];
    }

    T operator[](size_t index)const {
        return weights[index];
    }



    size_t getSize() {
        return SIZE;
    }

};


#endif //CHECKERENGINEX_WEIGHTS_H
