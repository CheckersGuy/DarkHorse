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

    std::array<std::array<T, 16>, 7> tempo_ranks;

    Weights() : kingOp(500), kingEnd(500) {
        weights = std::vector<T>(SIZE, T{0});
        for (auto i = 0; i < tempo_ranks.size(); ++i) {
            std::fill(tempo_ranks[i].begin(), tempo_ranks[i].end(), T{0});
        }
    }

    Weights(const Weights<T> &other) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::copy(other.weights.begin(), other.weights.end(), weights.begin());
        std::copy(tempo_ranks.begin(), tempo_ranks.end(), other.tempo_ranks.begin());
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
        if (!stream.good()) {
            std::cerr << "Error could not load the weights" << std::endl;
            return;
        }

        size_t counter = 0u;
        while (stream) {
            if (counter >= SIZE)
                break;
            RunType length;
            double first;
            stream.read((char *) &length, sizeof(RunType));
            stream.read((char *) &first, sizeof(double));
            if (stream.eof())
                break;
            for (RunType i = 0u; i < length; ++i) {
                weights[counter] = std::clamp(first, (double) std::numeric_limits<int16_t>::min(),
                                              (double) std::numeric_limits<int16_t>::max());
                counter++;
            }
        }
        double kingOpVal, kingEndVal;
        stream.read((char *) &kingOpVal, sizeof(double));
        stream.read((char *) &kingEndVal, sizeof(double));
        this->kingOp = static_cast<T>(kingOpVal);
        this->kingEnd = static_cast<T>(kingEndVal);

        for (auto i = 0; i < tempo_ranks.size(); ++i) {
            for (auto j = 0; j < 16; ++j) {
                double temp;
                stream.read((char *) &temp, sizeof(double));
                tempo_ranks[i][j] = std::clamp(temp, (double) std::numeric_limits<int16_t>::min(),
                                               (double) std::numeric_limits<int16_t>::max());
            }
        }
    }

    template<typename RunType=uint32_t>
    void load_weights(std::string input_file) {
        std::ifstream stream(input_file, std::ios::binary);
        load_weights(stream);
    }


    template<typename RunType=uint32_t>
    void store_weights(std::ofstream &stream) {
        using DataType = double;
        auto end = weights.end();
        for (auto it = weights.begin(); it != end;) {
            RunType length{0};
            auto first = *it;
            while (it != end && length < std::numeric_limits<RunType>::max() && *(it) == first) {
                length++;
                it++;
            }
            stream.write((char *) &length, sizeof(RunType));
            stream.write((char *) &first, sizeof(DataType));
        }
        stream.write((char *) &kingOp, sizeof(DataType));
        stream.write((char *) &kingEnd, sizeof(DataType));
        for (auto i = 0; i < tempo_ranks.size(); ++i) {
            for (auto j = 0; j < 16; ++j) {
                stream.write((char *) &tempo_ranks[i][j], sizeof(DataType));
            }
        }
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

        uint32_t man_black = pos.BP & (~pos.K);
        uint32_t man_white = pos.WP & (~pos.K);
        man_white = getMirrored(man_white);
        U tempi = 0;


        for (int i = 0; i < 7; ++i) {
            uint32_t shift = 4u * i;
            const uint32_t mask_white = (man_white >> shift) & temp_mask;
            const uint32_t mask_black = (man_black >> shift) & temp_mask;
            tempi -= tempo_ranks[i][mask_black];
            tempi += tempo_ranks[i][mask_white];
        }

        U phase = WP + BP;

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
            //FOR THE PROMO_SQUARES
            Bits::big_index(f, pos.WP, pos.BP, pos.K);

        } else {
            Bits::small_index(f, pos.WP, pos.BP, pos.K);
        }


        opening *= color;
        ending *= color;

        const U pieceEval = (WP - BP) * pawnEval;
        const U kingEvalOp = (pawnEval + kingOp) * (WK - BK);
        const U kingEvalEnd = (pawnEval + kingEnd) * (WK - BK);
        opening += kingEvalOp;
        opening += pieceEval;
        opening += tempi;

        ending += kingEvalEnd;
        ending += pieceEval;
        ending += tempi;

        U score = (phase * opening + (stage_size - phase) * ending);
        if constexpr(std::is_floating_point_v<U>) {
            score = score / stage_size;
        } else {
            score = div_round(score, stage_size);
        }
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

    Weights &operator=(const Weights &others) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::copy(others.weights.begin(), others.weights.end(), weights.begin());
        std::copy(tempo_ranks.begin(), tempo_ranks.end(), others.tempo_ranks.begin());
        this->kingOp = others.kingOp;
        this->kingEnd = others.kingEnd;
        return *this;
    }

};


#endif //CHECKERENGINEX_WEIGHTS_H
