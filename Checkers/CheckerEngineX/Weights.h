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
constexpr size_t powers[] = {1, 5, 25, 125, 625, 3125, 15625, 78125};

constexpr size_t SIZE = 390625 * 9  * 2 * 2;

inline size_t getIndex(uint32_t region, const Position &pos) {
    //will return the index for a given position
    size_t index = 0;
    uint32_t pieces = region & (pos.BP | pos.WP);
    size_t counter = 0;
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

    Weights(){
        weights = new T[SIZE];
        std::memset(weights, 0, sizeof(T) * SIZE);
    }

    Weights(const Weights& other){
        weights = new T[SIZE];
        std::memcpy(this->weights,other.weights,sizeof(T)*SIZE);
    }

    Weights(Weights&& other){
        this->weights=other.weights;
        other.weights= nullptr;
    }

    ~Weights() {
        delete[] weights;
    }

    int numNonZeroValues() {
        int counter = 0;
        for (size_t i = 0; i < SIZE; ++i) {
            if (((char)(round(weights[i]))) != 0) {
                counter++;
            }
        }
        return counter;
    }

    T getMaxValue() {
        T tempValue = -10000;
        for (size_t i = 0; i < SIZE; ++i) {
            if (weights[i] > tempValue) {
                tempValue = weights[i];
            }
        }
        return tempValue;
    }

    T getMinValue() {
        T tempValue = 10000;
        for (size_t i = 0; i < SIZE; ++i) {
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
            size_t counter=0;
            while(!stream.eof()){
                uint32_t runLength=0;
                stream.read((char*)&runLength,sizeof(uint32_t));


                char value;
                stream.read(&value,sizeof(char));
                for(size_t i=0;i<runLength;++i){
                    weights[counter]=(T)(value);
                    counter++;
                }
            }
        }
        stream.close();
    }


    void storeWeights(const std::string path) {
        std::ofstream stream(path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Error" << std::endl;
            exit(0);
        }
        for(size_t i=0;i<SIZE;++i){
            uint32_t runLength=1;
            char value =(char)round(weights[i]);
            while(i<SIZE && ((char)round(weights[i]))==((char)round(weights[i+1]))){
                ++i;
                ++runLength;
            }
            stream.write((char*)&runLength,sizeof(uint32_t));
            stream.write(&value,sizeof(char));
        }
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

        size_t colorIndex = 0;
        if (pos.color == WHITE) {
            colorIndex = 390625 * 9 * 2;
        }

        for (size_t j = 0; j < 3; ++j) {
            for (size_t i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8 * j + i);
                size_t index = getIndex(curRegion, pos)+colorIndex;
                if constexpr(std::is_same<char,T>::value){
                    opening +=  (weights[index + 390625 * (3 * j + i)]);
                    ending +=  (weights[index + 390625 * 9 + 390625 * (3 * j + i)]);
                }else{
                    opening +=  (int)(weights[index + 390625 * (3 * j + i)]);
                    ending +=  (int)(weights[index + 390625 * 9 + 390625 * (3 * j + i)]);
                }
            }
        }

        int phasedScore = 0;
        phasedScore += ((phase * opening + (24 - phase) * ending));
        phasedScore += ((phase * openingWhite + (24 - phase) * endingWhite));
        phasedScore /= 24;
        phasedScore -= ((phase * openingBlack + (24 - phase) * endingBlack)) / 24;
        sum += phasedScore;

        int balanceWhite =std::abs(leftWhite-rightWhite);
        int balanceBlack =std::abs(leftBlack-rightBlack);
        sum+=2*(balanceBlack-balanceWhite);



        return sum;
    }

    size_t getSize() {
        return SIZE ;
    }

};



#endif //CHECKERENGINEX_WEIGHTS_H
