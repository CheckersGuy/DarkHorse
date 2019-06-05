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
constexpr size_t powers[] = {1, 5, 25, 125, 625, 3125, 15625, 78125};

constexpr size_t SIZE = 390625*9*2;



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
struct Weights {

    //scalFac is used to scale the evalTerms pieceEval and maybe others

    T kingOp, kingEnd;
    T balanceOp, balanceEnd;

    std::unique_ptr<T[]> weights;

    Weights() : kingOp(0), kingEnd(0), balanceOp(0), balanceEnd(0) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::memset(weights.get(), 0, sizeof(T) * SIZE);
    }

    Weights(const Weights<T> &other) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::memcpy(this->weights.get(), other.weights.get(), sizeof(T) * SIZE);
    }

    Weights(Weights &&other) {
        this->weights = other.weights;
        other.weights = nullptr;
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

    T getMaxValue()const {
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
            while (!stream.eof() && counter<SIZE) {
                uint32_t runLength = 0;
                stream.read((char *) &runLength, sizeof(uint32_t));
                double value;
                stream.read((char *) &value, sizeof(double));
                for (size_t i = 0; i < runLength; ++i) {
                    weights[counter] = value*scalFac;
                    counter++;
                }
            }
            for(size_t i=SIZE;i<SIZE+4;++i){
                double current;
                stream.read((char*)&current,sizeof(double));
                (*this)[i]=current*scalFac;

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


        stream.write((char*)&kingOp,sizeof(T));
        stream.write((char*)&kingEnd,sizeof(T));
        stream.write((char*)&balanceOp,sizeof(T));
        stream.write((char*)&balanceEnd,sizeof(T));

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

        int sum = 100*scalFac * (WP - BP);
        int phase = WP + BP;
        int WK = 0;
        int BK = 0;
        if (pos.K != 0) {
            WK = __builtin_popcount(pos.K & (pos.WP));
            BK = __builtin_popcount(pos.K & (pos.BP));
            phase += WK + BK;
        }

        int kingEval = 0;
        int balanceScore = std::abs(leftBlack - rightBlack) - std::abs(leftWhite - rightWhite);

        constexpr T div= static_cast<T>(24);

        T factorOp = phase;
        T factorEnd = div - phase;
        T temp = (factorOp*static_cast<T>(100*scalFac+kingOp)+factorEnd*static_cast<T>(100*scalFac+kingEnd ))/div;
        T temp2 = (factorOp*balanceOp + factorEnd*balanceEnd)/div ;
        if constexpr(std::is_same<T, double>::value) {
            kingEval += static_cast<int>(temp)* (WK - BK) + temp2*balanceScore;
        } else if constexpr(std::is_same<T, int>::value) {
            kingEval += temp  * (WK - BK) + temp2*balanceScore;
        }


        if (pos.getColor() == BLACK) {
            pos = pos.getColorFlip();
        }
        T opening = 0, ending = 0;

        for(int j=0;j<3;++j){

        for (int i = 0; i < 3; ++i) {
            const uint32_t curRegion = region << (8 * j + i);
            size_t indexOpening = 18 * getIndex(curRegion, pos) + 3 * j + i;
            size_t indexEnding = 18 * getIndex(curRegion, pos) + 9 + 3 * j + i;
            opening += (weights[indexOpening]) * static_cast<T>(color);
            ending += (weights[indexEnding] ) * static_cast<T>(color);
        }
        }

        T opFactor = phase;
        T endFactor = div - phase;
        T phaseTemp = phase;


        T phasedPatterns = (opFactor * opening + endFactor * ending)/div;
        sum += phasedPatterns;
        sum += kingEval;

        return sum;
    }

    T &operator[](size_t index) {
        if (index == SIZE) {
            return kingOp;
        } else if (index == SIZE + 1) {
            return kingEnd;
        } else if (index == SIZE + 2) {
            return balanceOp;
        } else if (index == SIZE + 3) {
            return balanceEnd;
        }
        return weights[index];
    }

    T operator[](size_t index) const {
        if (index == SIZE) {
            return kingOp;
        } else if (index == SIZE + 1) {
            return kingEnd;
        } else if (index == SIZE + 2) {
            return balanceOp;
        } else if (index == SIZE + 3) {
            return balanceEnd;
        }

        return weights[index];
    }


    size_t getSize() const {
        return SIZE;
    }

};


#endif //CHECKERENGINEX_WEIGHTS_H
