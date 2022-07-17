//
// Created by robin on 7/30/18.
//

#ifndef CHECKERENGINEX_WEIGHTS_H
#define CHECKERENGINEX_WEIGHTS_H

#include <string>
#include "Bits.h"
#include "Position.h"
#include <fstream>
#include <iomanip>
#include "MGenerator.h"
#include "GameLogic.h"
#include <cstring>


constexpr size_t SIZE = 18ull * 390625ull + 4ull * 531441ull + 8ull * 157464ull;


template<typename T>
struct Weights {
    T kingOp, kingEnd;
    std::vector<T> weights;

    Weights() : kingOp(500), kingEnd(500) {
        weights = std::vector<T>(SIZE, T{0});
    }

    Weights(const Weights<T> &other) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::copy(other.weights.begin(), other.weights.end(), weights.begin());
        this->kingOp = other.kingOp;
        this->kingEnd = other.kingEnd;
    }


    size_t num_non_zero_weights() {
        return std::count_if(weights.begin(), weights.end(), [](T val) { return static_cast<int>(val) != 0; });
    }

    T get_max_weight() const {
        return *std::max_element(weights.begin(), weights.end());
    }

    T get_min_weight() const {
        return *std::min_element(weights.begin(), weights.end());
    }


    template<typename RunType=uint32_t>
    void load_weights(std::ifstream &stream) {
        static_assert(std::is_unsigned<RunType>::value);

            if(!stream.good()){
            std::cerr<<"Could not load the weights"<<std::endl;
            std::exit(-1);
        }
        if (!stream.good()) {
            std::cerr << "Error could not load the weights" << std::endl;
            return;
        }
        
        for(auto i=0;i<weights.size();++i){
            double value;
            stream.read((char*)&value,sizeof(double));
            weights[i]=std::round(value);
        }
        double kingOpVal,kingEndVal;
        stream.read((char *) &kingOpVal, sizeof(double));
        stream.read((char *) &kingEndVal, sizeof(double));
        this->kingOp = static_cast<T>(kingOpVal);
        this->kingEnd = static_cast<T>(kingEndVal);  
    }

    template<typename RunType=uint32_t>
    void load_weights(std::string input_file) {
        std::ifstream stream(input_file, std::ios::binary);
        load_weights(stream);
    }


    template<typename RunType=uint32_t>
    void store_weights(std::ofstream &stream) {
        using DataType = double;
          for(auto i=0;i<weights.size();++i){
            stream.write((char*)&weights[i],sizeof(double));
        }
         stream.write((char *) &kingOp, sizeof(double));
         stream.write((char *) &kingEnd, sizeof(double));
    }

    template<typename RunType=uint32_t>
    void store_weights(const std::string &path) {
        std::ofstream stream(path, std::ios::binary);
        store_weights<RunType>(stream);
    }


    template<typename U=int32_t>
    U evaluate(Position pos, int ply) const {
        if (pos.BP == 0) {
            return -loss(ply);
        }
        if (pos.WP == 0) {
            return loss(ply);
        }

        const U color = pos.color;
        constexpr U pawnEval = 0;
        const U WP = Bits::pop_count(pos.WP & (~pos.K));
        const U BP = Bits::pop_count(pos.BP & (~pos.K));
        U phase = BP+WP;

        U WK = 0;
        U BK = 0;
        if (pos.K != 0) {
            WK = Bits::pop_count(pos.WP & pos.K);
            BK = Bits::pop_count(pos.BP & pos.K);
            phase += WK + BK;
        }
        U opening = 0, ending = 0;

        if (pos.get_color() == BLACK) {
            pos = pos.get_color_flip();
        } 
        auto f = [&](size_t op_index) {
            size_t end_index = op_index + 1;
            opening += weights[op_index];
            ending += weights[end_index];
        };
        if (pos.K == 0) {
            Bits::big_index(f, pos.WP, pos.BP, pos.K);
        } else {
            Bits::small_index(f, pos.WP, pos.BP, pos.K);
        }


        opening *= color;
        ending *= color;

       
        const U pieceEval = (WP-BP)*100;
        const U kingEvalOp = (kingOp) * (WK - BK);
        const U kingEvalEnd = (kingEnd) * (WK - BK);
        opening += kingEvalOp;
        opening += pieceEval;
        ending += kingEvalEnd;
        ending += pieceEval;

        const U stage_size = U{24};
        U score = (phase * opening + (stage_size - phase) * ending);
        score = score / stage_size;

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
    //needs some serious reworking
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

    Weights &operator=(const Weights &others) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::copy(others.weights.begin(), others.weights.end(), weights.begin());
        this->kingOp = others.kingOp;
        this->kingEnd = others.kingEnd;
        return *this;
    }

};


#endif //CHECKERENGINEX_WEIGHTS_H
