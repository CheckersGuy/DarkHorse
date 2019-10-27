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


constexpr uint32_t region = 13107u;
constexpr std::array<uint64_t, 8> powers = {1ull, 5ull, 25ull, 125ull, 625ull, 3125ull, 15625ull, 78125ull};
constexpr size_t SIZE = 390625ull * 9ull * 2ull;

inline size_t getIndex(uint32_t reg, const Position &pos) {
    //will return the index for a given position
    size_t index = 0ull;
    uint32_t pieces = reg & (pos.BP | pos.WP);
    size_t counter = 0ull;
    uint32_t BP = pos.BP & (~pos.K);
    uint32_t WP = pos.WP & (~pos.K);
    uint32_t BK = pos.BP & pos.K;
    uint32_t WK = pos.WP & pos.K;
    while (pieces) {
        uint32_t lsb = (pieces & ~(pieces - 1u));
        pieces &= pieces - 1u;
        size_t current;
        current = ((BP & lsb) != 0) * 3 + ((WP & lsb) != 0) * 4 + ((BK & lsb) != 0) * 1 + ((WK & lsb) != 0) * 2;
        index += powers[counter++] * current;

    }
    return index;
}

template<typename T>
struct Weights {


    T kingOp, kingEnd;
    std::unique_ptr<T[]> weights;

    Weights() : kingOp(0), kingEnd(0) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::memset(weights.get(), 0, sizeof(T) * SIZE);
    }

    Weights(const Weights<T> &other) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::memcpy(this->weights.get(), other.weights.get(), sizeof(T) * SIZE);
    }


    size_t numNonZeroValues() {
        return std::count_if(weights.get(), weights.get() + SIZE, [](T val) { return static_cast<int>(val) != 0; });
    }

    T getNorm() const {
        T current = 0;
        for (size_t i = 0; i < SIZE; ++i) {
            current += weights[i] * weights[i];
        }
        current = std::sqrt(current);
        return current;
    }

    T getMaxValue() const {
        return *std::max_element(weights.get(), weights.get() + SIZE);
    }

    T getMinValue() const {
        return *std::min_element(weights.get(), weights.get() + SIZE);
    }

    void loadWeights(const std::string path) {
        std::ifstream stream(path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Error: Couldn't find weights, default init" << std::endl;;
        } else {
            size_t counter = 0;
            while (!stream.eof() && counter < SIZE) {
                uint32_t runLength = 0;
                stream.read((char *) &runLength, sizeof(uint32_t));
                double value;
                stream.read((char *) &value, sizeof(double));
                for (size_t i = 0; i < runLength; ++i) {
                    weights[counter] = value;
                    counter++;
                }
            }
            for (size_t i = SIZE; i < SIZE + 2; ++i) {
                double current;
                stream.read((char *) &current, sizeof(double));
                (*this)[i] = current;

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


        stream.write((char *) &kingOp, sizeof(T));
        stream.write((char *) &kingEnd, sizeof(T));
        stream.close();
    }

    Value evaluate(Position pos) const {
        constexpr int pawnEval = 100 * scalFac;
        const Color color = pos.getColor();
        const uint32_t nKings = ~pos.K;

        const int WP = __builtin_popcount(pos.WP & (~pos.K));
        const int BP = __builtin_popcount(pos.BP & (~pos.K));


        int phase = WP + BP;
        int WK = 0;
        int BK = 0;
        if (pos.K != 0) {
            WK = __builtin_popcount(pos.WP & pos.K);
            BK = __builtin_popcount(pos.BP & pos.K);
            phase += WK + BK;
        }


        if (pos.getColor() == BLACK) {
            pos = pos.getColorFlip();
        }
        int opening = 0, ending = 0;

        for (int j = 0; j < 3; ++j) {
            for (int i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8u * j + i);
                size_t indexOpening = 18 * getIndex(curRegion, pos) + 3 * j + i;
                size_t indexEnding = 18 * getIndex(curRegion, pos) + 9 + 3 * j + i;
                opening += (weights[indexOpening]);
                ending += (weights[indexEnding]);
            }
        }
        opening *= color;
        ending *= color;
        const int factorOp = phase;
        const int factorEnd = 24 - phase;


        const int pieceEval = (WP - BP) * pawnEval;
        const int kingEvalOp = (pawnEval + kingOp) * (WK - BK);
        const int kingEvalEnd = (pawnEval + kingEnd) * (WK - BK);


        opening += kingEvalOp;
        opening += pieceEval;

        ending += kingEvalEnd;
        ending += pieceEval;

        int score = (factorOp * opening + factorEnd * ending) / 24;


        return score;
    }

    T &operator[](size_t index) {
        if (index == SIZE) {
            return kingOp;
        } else if (index == SIZE + 1) {
            return kingEnd;
        }
        return weights[index];
    }

    const T &operator[](size_t index) const {
        if (index == SIZE) {
            return kingOp;
        } else if (index == SIZE + 1) {
            return kingEnd;
        }
        return weights[index];
    }


    size_t getSize() const {
        return SIZE;
    }
};

#ifdef TRAIN
extern Weights<double> gameWeights;
#else
extern Weights<int> gameWeights;
#endif

#endif //CHECKERENGINEX_WEIGHTS_H
