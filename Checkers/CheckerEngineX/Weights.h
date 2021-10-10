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

enum class IndexType {
    INNER, PROMO
};

//constexpr uint32_t region = 13107u;
//constexpr size_t SIZE = 12ull * 531441ull + 24ull * 15625ull;//changing how indices are calculated
//constexpr size_t SIZE = 12ull * 531441ull + 24ull * 390625ull;
//constexpr size_t SIZE = 18ull * 390625ull + 12ull * 531441ull;

constexpr size_t SIZE = 18ull * 390625ull + 4ull * 531441ull + 8ull * 157464ull;

template<IndexType type>
size_t get_big_index(uint32_t reg, const Position &pos) {
    const uint32_t PROMO_SQUARES = PROMO_SQUARES_BLACK | PROMO_SQUARES_WHITE;
    size_t index = 0ull;
    uint32_t BP = pos.BP & (~pos.K);
    uint32_t WP = pos.WP & (~pos.K);
    uint32_t orig_pieces = (pos.BP | pos.WP) & reg & (~PROMO_SQUARES);
    uint32_t promo_region = reg & PROMO_SQUARES;

    uint32_t promo_pieces = (pos.BP | pos.WP) & PROMO_SQUARES & reg;
    uint32_t promo_index = Bits::pext(promo_pieces, promo_region);
    uint32_t pieces = Bits::pext(orig_pieces, reg & (~PROMO_SQUARES));

    while (pieces) {
        uint32_t lsb = (orig_pieces & ~(orig_pieces - 1u));
        size_t temp_index = Bits::bitscan_foward(pieces);
        size_t current = ((BP & lsb) != 0u) * 1ull + ((WP & lsb) != 0u) * 2ull;
        index += current * powers3[temp_index];
        pieces &= pieces - 1u;
        orig_pieces &= orig_pieces - 1u;
    }

    if constexpr(type == IndexType::INNER) {
        return index;
    } else {
        return 8 * index + promo_index;
    }
}


inline size_t getIndex2(uint32_t reg, const Position &pos) {
    uint32_t orig_pieces = (pos.BP | pos.WP) & reg;
    uint32_t pieces = (pos.BP | pos.WP);
    pieces = Bits::pext(pieces, reg);

    uint32_t BP = pos.BP & (~pos.K);
    uint32_t WP = pos.WP & (~pos.K);
    uint32_t BK = pos.BP & pos.K;
    uint32_t WK = pos.WP & pos.K;
    size_t index = 0ull;
    while (orig_pieces) {
        uint32_t lsb = (orig_pieces & ~(orig_pieces - 1u));
        size_t temp_index = Bits::bitscan_foward(pieces);
        size_t current = ((BP & lsb) != 0u) * 1ull + ((WP & lsb) != 0u) * 2ull + ((BK & lsb) != 0u) * 3ull +
                         ((WK & lsb) != 0u) * 4ull;

        index += current * powers5[temp_index];
        pieces &= pieces - 1u;
        orig_pieces &= orig_pieces - 1u;
    }

    return index;
}

template<typename T>
struct Weights {
    T kingOp, kingEnd;
    std::vector<T> weights;

    std::array<std::array<T, 16>, 7> tempo_ranks;

    Weights() : kingOp(0), kingEnd(0) {
        weights = std::vector<T>(SIZE, T{0});
        for (auto i = 0; i < tempo_ranks.size(); ++i) {
            std::fill(tempo_ranks[i].begin(), tempo_ranks[i].end(), T{0});
        }
    }

    Weights(const Weights<T> &other) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::copy(other.weights.begin(), other.weights.end(), weights.begin());
        this->kingOp = other.kingOp;
        this->kingOp = other.kingOp;
    }


    size_t numNonZeroValues() {
        return std::count_if(weights.begin(), weights.end(), [](T val) { return static_cast<int>(val) != 0; });
    }

    T getMaxValue() const {
        return *std::max_element(weights.begin(), weights.end());
    }

    T getMinValue() const {
        return *std::min_element(weights.begin(), weights.end());
    }


    template<typename RunType=uint32_t>
    void loadWeights(const std::string &path) {
        static_assert(std::is_unsigned<RunType>::value);
        std::ifstream stream(path, std::ios::binary);
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
        stream.close();
    }

    template<typename RunType=uint32_t>
    void storeWeights(const std::string &path) const {
        using DataType = double;
        std::ofstream stream(path, std::ios::binary);
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
        stream.close();
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
        constexpr U pawnEval = 1000;
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

        if (pos.getColor() == BLACK) {
            pos = pos.getColorFlip();
        }
        const size_t offset1 = 8ull * 157464ull;
        const size_t offset2 = 4ull * 531441ull + 8ull * 157464ull;
        if (pos.K == 0) {
            //FOR THE PROMO_SQUARES
            for (auto i: {0, 2}) {
                size_t temp = ((i == 0) ? 0 : 1);
                for (auto k = 0; k < 2; ++k) {
                    Position test;
                    const uint32_t sub_reg = big_region << (8 * i + k);
                    test.BP = sub_reg;
                    size_t index = get_big_index<IndexType::PROMO>(sub_reg, pos);
                    size_t sub_index_op = 8 * index + 2 * k + 4 * temp;
                    size_t sub_index_end = 8 * index + 2 * k + 4 * temp + 1;
                    opening += weights[sub_index_op];
                    ending += weights[sub_index_end];
                }
            }
            //FOR THE NON_PROMO_SQUARES
            for (auto k = 0; k < 2; ++k) {
                const uint32_t sub_reg = big_region << (8 * 1 + k);
                size_t index = get_big_index<IndexType::INNER>(sub_reg, pos);
                size_t sub_index_op = 4 * index + 2 * k;
                size_t sub_index_end = 4 * index + 2 * k + 1;
                opening += weights[sub_index_op + offset1];
                ending += weights[sub_index_end + offset1];
            }

        } else {
            for (auto i = 0; i < 3; ++i) {
                for (auto k = 0; k < 3; ++k) {
                    const uint32_t sub_reg = region << (8 * i + k);
                    size_t index = getIndex2(sub_reg, pos);
                    size_t sub_index_op = 18 * index + 2 * k + 6 * i;
                    size_t sub_index_end = 18 * index + 2 * k + 6 * i + 1;
                    opening += weights[sub_index_op + offset2];
                    ending += weights[sub_index_end + offset2];
                }
            }

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
};

#ifdef TRAIN
extern Weights<double> gameWeights;
#else
extern Weights<int16_t> gameWeights;

#endif

#endif //CHECKERENGINEX_WEIGHTS_H
