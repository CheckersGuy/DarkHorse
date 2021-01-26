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
#include <iterator>

constexpr uint32_t region = 13107u;
constexpr size_t SIZE = 390625ull * 9ull * 2ull;

inline size_t getIndex2(uint32_t reg, const Position &pos) {
    //trying to be a little bit more efficient with the pext instruction
    uint32_t orig_pieces = (pos.BP | pos.WP) & reg;
    uint32_t pieces = (pos.BP | pos.WP);
    pieces = _pext_u32(pieces, reg);

    uint32_t BP = pos.BP & (~pos.K);
    uint32_t WP = pos.WP & (~pos.K);
    uint32_t BK = pos.BP & pos.K;
    uint32_t WK = pos.WP & pos.K;
    size_t index = 0ull;
    while (pieces) {
        uint32_t lsb = (orig_pieces & ~(orig_pieces - 1u));
        size_t temp_index = Bits::bitscan_foward(pieces);
        size_t current = ((BP & lsb) != 0u) * 3ull + ((WP & lsb) != 0u) * 4ull + ((BK & lsb) != 0u) * 1ull +
                         ((WK & lsb) != 0u) * 2ull;
        index += current * powers[temp_index];
        pieces &= pieces - 1u;
        orig_pieces &= orig_pieces - 1u;
    }

    return index;
}

template<typename T>
struct Weights {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    T kingOp, kingEnd;
    std::unique_ptr<T[]> weights;
    std::array<std::array<T, 16>, 7> tempo_ranks;

    Weights() : kingOp(1500), kingEnd(1500), weights(std::make_unique<T[]>(SIZE)) {
        std::fill(weights.get(), weights.get() + SIZE, T{0});
        for (auto i = 0; i < tempo_ranks.size(); ++i) {
            std::fill(tempo_ranks[i].begin(), tempo_ranks[i].end(), T{0});
        }
    }

    Weights(const Weights<T> &other) {
        this->weights = std::make_unique<T[]>(SIZE);
        std::copy(other.weights.get(), other.weights.get() + SIZE, weights.get());
        this->kingOp = other.kingOp;
        this->kingOp = other.kingOp;
    }


    size_t numNonZeroValues() {
        return std::count_if(weights.get(), weights.get() + SIZE, [](T val) { return static_cast<int>(val) != 0; });
    }

    T getMaxValue() const {
        return *std::max_element(weights.get(), weights.get() + SIZE);
    }

    T getMinValue() const {
        return *std::min_element(weights.get(), weights.get() + SIZE);
    }

    int king_mobility(Position &pos) const {
        uint32_t kings_white = pos.WP & pos.K;
        uint32_t kings_black = pos.BP & pos.K;
        int count_denied = 0;
        int count_safe = 0;

        if (kings_white != 0u) {
            uint32_t attacked = pos.attacks<BLACK>();
            while (kings_white != 0u) {
                uint32_t bit_mask = kings_white & (~(kings_white - 1u));
                uint32_t squares = pos.getKingAttackSquares(bit_mask);
                uint32_t safe_squares = squares & (~attacked);
                uint32_t denied_squares = squares & attacked;
                count_safe += __builtin_popcount(safe_squares);
                count_denied += __builtin_popcount(denied_squares);
                kings_white &= kings_white - 1u;
            }
        }

        if (kings_black != 0u) {
            uint32_t attacked = pos.attacks<WHITE>();
            while (kings_black != 0u) {
                uint32_t bit_mask = kings_black & (~(kings_black - 1u));
                uint32_t squares = pos.getKingAttackSquares(bit_mask);
                uint32_t safe_squares = squares & (~attacked);
                uint32_t denied_squares = squares & attacked;
                count_safe -= __builtin_popcount(safe_squares);
                count_denied -= __builtin_popcount(denied_squares);
                kings_black &= kings_black - 1u;
            }
        }
        return count_denied;
    }


    template<typename RunType=uint32_t>
    void loadWeights(const std::string &path) {
        using DataType = double;
        static_assert(std::is_unsigned<RunType>::value);
        std::ifstream stream(path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Error" << std::endl;
            return;
        }

        size_t counter = 0u;
        while (stream) {
            if (counter >= SIZE)
                break;
            RunType length;
            DataType first;
            stream.read((char *) &length, sizeof(RunType));
            stream.read((char *) &first, sizeof(DataType));
            if (stream.eof())
                break;
            for (RunType i = 0u; i < length; ++i) {
                weights[counter] = first;
                counter++;
            }
        }
        DataType kingOpVal, kingEndVal;
        stream.read((char *) &kingOpVal, sizeof(DataType));
        stream.read((char *) &kingEndVal, sizeof(DataType));
        this->kingOp = static_cast<T>(kingOpVal);
        this->kingEnd = static_cast<T>(kingEndVal);

        for (auto i = 0; i < tempo_ranks.size(); ++i) {
            for (auto j = 0; j < 16; ++j) {
                double temp;
                stream.read((char *) &temp, sizeof(DataType));
                tempo_ranks[i][j] = temp;
            }
        }
        stream.close();
    }

    template<typename RunType=uint32_t>
    void storeWeights(const std::string &path) const {
        using DataType = double;
        std::ofstream stream(path, std::ios::binary);
        auto end = weights.get() + SIZE;
        for (auto it = weights.get(); it != end;) {
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
        U color = pos.getColor();
        constexpr U pawnEval = 1000;
        const U WP = Bits::pop_count(pos.WP & (~pos.K));
        const U BP = Bits::pop_count(pos.BP & (~pos.K));

        uint32_t man_black = pos.BP & (~pos.K);
        uint32_t man_white = pos.WP & (~pos.K);
        man_white = getMirrored(man_white);
        U tempi = 0;
        for (uint32_t i = 0; i < 7; ++i) {
            uint32_t shift = 4u * i;
            auto mask_white = (man_white >> shift) & temp_mask;
            auto mask_black = (man_black >> shift) & temp_mask;
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
        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8u * j + i);
                const auto region_index = getIndex2(curRegion, pos);
                size_t indexOpening = 18u * region_index + 3u * j + i;
                size_t indexEnding = 18u * region_index + 9u + 3u * j + i;
                opening += (weights[indexOpening]);
                ending += (weights[indexEnding]);
            }
        }

        const U pieceEval = (WP - BP) * pawnEval;
        const U kingEvalOp = kingOp * (WK - BK);
        const U kingEvalEnd = kingEnd * (WK - BK);
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
            score = div_round(score, (int) stage_size);
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

    T averageWeight() const {
        T accumulation = 0;
        T counter = 0;
        std::for_each(weights.get(), weights.get() + SIZE, [&](T value) {
            if (value != 0) {
                counter++;
                accumulation += std::abs(value);
            }
        });
        return accumulation / counter;
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
