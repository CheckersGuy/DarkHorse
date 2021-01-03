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
#include <filesystem>

constexpr uint32_t region = 13107u;
constexpr size_t SIZE = 390625ull * 9ull * 2ull;
constexpr std::array<size_t, 8> powers = {1ull, 5ull, 25ull, 125ull, 625ull, 3125ull, 15625ull, 78125ull};
namespace fs = std::filesystem;


inline std::optional<fs::path> has_directory(const fs::path &path, std::string directory_name) {
    if (fs::exists(path) && fs::is_directory(path)) {
        for (const auto &entry : fs::directory_iterator(path)) {
            if (fs::is_directory(entry.path()) && entry.path().filename() == directory_name)
                return std::make_optional(entry.path());
        }
    }
    return std::nullopt;
}

inline std::optional<fs::path> has_file(const fs::path &path, std::string file_name) {
    if (fs::exists(path) && fs::is_directory(path)) {
        for (const auto &entry: fs::directory_iterator(path)) {
            if (fs::is_regular_file(entry.path()) && entry.path().filename() == file_name)
                return std::make_optional(entry.path());
        }
    }
    return std::nullopt;
}


inline size_t getIndex(uint32_t reg, const Position &pos) {
    //will return the index for a given position

    uint32_t pieces = (pos.BP | pos.WP);
    pieces = _pext_u32(pieces, reg);
    uint32_t BP = pos.BP & (~pos.K);
    BP = _pext_u32(BP, reg);
    uint32_t WP = pos.WP & (~pos.K);
    WP = _pext_u32(WP, reg);
    uint32_t BK = pos.BP & pos.K;
    BK = _pext_u32(BK, reg);
    uint32_t WK = pos.WP & pos.K;
    WK = _pext_u32(WK, reg);
    size_t index = 0ull;
    while (pieces) {
        uint32_t lsb = (pieces & ~(pieces - 1u));
        size_t temp_index = Bits::bitscan_foward(lsb);
        size_t current = ((BP & lsb) != 0u) * 3ull + ((WP & lsb) != 0u) * 4ull + ((BK & lsb) != 0u) * 1ull +
                         ((WK & lsb) != 0u) * 2ull;
        index += current * powers[temp_index];
        pieces &= pieces - 1u;
    }

    return index;
}

inline size_t getIndex2(uint32_t reg, const Position &pos,uint32_t shift) {
    //will return the index for a given position


    uint32_t pieces = reg&(pos.BP | pos.WP);
    pieces =pieces>>shift;
    uint32_t BP = reg&(pos.BP & (~pos.K));
    BP=BP>>shift;
    uint32_t WP = reg&(pos.WP & (~pos.K));
    WP=WP>>shift;
    uint32_t BK = reg&(pos.BP & pos.K);
    BK=BK>>shift;
    uint32_t WK = reg&(pos.WP & pos.K);
    WK=WK>>shift;
    size_t index = 0ull;
    while (pieces) {
        uint32_t lsb = (pieces & ~(pieces - 1u));
        size_t temp_index = Bits::bitscan_foward(lsb);
        size_t current = ((BP & lsb) != 0u) * 3ull + ((WP & lsb) != 0u) * 4ull + ((BK & lsb) != 0u) * 1ull +
                         ((WK & lsb) != 0u) * 2ull;
        index += current * powers[temp_index];
        pieces &= pieces - 1u;
    }

    return index;
}

template<typename T>
struct Weights {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    T kingOp, kingEnd;
    std::unique_ptr<T[]> weights;
    std::array<std::array<T, 16>, 7> tempo_ranks;

    T balance;

    Weights() : kingOp(150), kingEnd(150), balance(-10), weights(std::make_unique<T[]>(SIZE)) {
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

    T getNorm() const {
        T current = std::transform_reduce(weights.get(), weights.get() + SIZE, weights.get(), 0.0,
                                          [](T v1, T v2) { return v1 + v2; }, [](T v1) { return v1 * v1; });
        return std::sqrt(current);
    }

    T getMaxValue() const {
        return *std::max_element(weights.get(), weights.get() + SIZE);
    }

    T getMinValue() const {
        return *std::min_element(weights.get(), weights.get() + SIZE);
    }

    int tempi_eval(Position &pos) const {
        uint32_t man = pos.BP & (~pos.K);
        auto tempi = -tempo_ranks[0][(man) & temp_mask];
        tempi -= tempo_ranks[1][(man >> 4u) & temp_mask];
        tempi -= tempo_ranks[2][(man >> 8u) & temp_mask];
        tempi -= tempo_ranks[3][(man >> 12u) & temp_mask];
        tempi -= tempo_ranks[4][(man >> 16u) & temp_mask];
        tempi -= tempo_ranks[5][(man >> 20u) & temp_mask];
        tempi -= tempo_ranks[6][(man >> 24u) & temp_mask];

        man = pos.WP & (~pos.K);
        man = getVerticalFlip(man);
        tempi += tempo_ranks[0][(man) & temp_mask];
        tempi += tempo_ranks[1][(man >> 4u) & temp_mask];
        tempi += tempo_ranks[2][(man >> 8u) & temp_mask];
        tempi += tempo_ranks[3][(man >> 12u) & temp_mask];
        tempi += tempo_ranks[4][(man >> 16u) & temp_mask];
        tempi += tempo_ranks[5][(man >> 20u) & temp_mask];
        tempi += tempo_ranks[6][(man >> 24u) & temp_mask];
        return tempi;
    }

    template<Color color>
    int balance_score(Position &pos) const {
        int num = 0;
        uint32_t men = pos.getCurrent<color>() & (~pos.K);
        int score = 0;
        for (auto i = 0; i < 8; ++i) {
            uint32_t column = columns[i];
            int num_pieces = __builtin_popcount(column & men);
            num += num_pieces;
            score += (i + 1) * num_pieces;
        }

        if (num == 0)
            return 0;

        score = score / num;
        return std::abs(4 - score);
    }

    int king_mobiility(Position &pos) const {
        //I may replace this with a simple table approach
        //where high values denote pieces a king should be on
        //I might remove denied squares
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

        return -count_denied + 2 * count_safe;

    }

    template<typename RunType=uint32_t>
    void loadWeights(const std::string &path) {
        using DataType = double;
        static_assert(std::is_unsigned<RunType>::value);
        fs::path home_path(getenv("HOME"));
        home_path /= "Dokumente";

        auto w_direc = has_directory(home_path, "CWeights");
        if (!w_direc.has_value()) {
            const fs::path tmp_path("/tmp");
            w_direc = has_directory(tmp_path, "CWeights");
        }
        if (!w_direc.has_value()) {
            std::cerr << "Could not find the weights_folder" << std::endl;
            return;
        }
        auto weight_path = (w_direc.has_value()) ? has_file(w_direc.value(), path) : std::nullopt;
        if (!weight_path.has_value()) {
            std::cerr << "Could not load the weights";
            return;
        }
        std::string path_string = weight_path.value().string();

        std::ifstream stream(path_string, std::ios::binary);
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
                weights[counter] = std::round(first);
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
                tempo_ranks[i][j] = std::round(temp);
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

    Value evaluate(Position pos, int ply) const {

        if (pos.isEnd()) {
            return pos.getColor() * loss(ply);
        }

        constexpr int pawnEval = 100;
        const Color color = pos.getColor();
        const int WP = __builtin_popcount(pos.WP & (~pos.K));
        const int BP = __builtin_popcount(pos.BP & (~pos.K));


        int tpo = tempi_eval(pos);
        //int mobility = king_mobiility(pos);
        int bala = (balance_score<WHITE>(pos) - balance_score<BLACK>(pos)) * balance;

        int phase = WP + BP;

        int WK = 0;
        int BK = 0;
        if (pos.K != 0) {
            WK = __builtin_popcount(pos.WP & pos.K);
            BK = __builtin_popcount(pos.BP & pos.K);
            phase += 2 * (WK + BK);
        }


        if (pos.getColor() == BLACK) {
            pos = pos.getColorFlip();
        }
        int opening = 0, ending = 0;

        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8u * j + i);
            /*    if(getIndex2(curRegion,pos,(8u * j + i))!=getIndex(curRegion,pos)){
                    std::cout<<"Error"<<std::endl;
                }*/
                size_t indexOpening = 18u * getIndex(curRegion, pos) + 3u * j + i;
                size_t indexEnding = 18u * getIndex(curRegion, pos) + 9u + 3u * j + i;
                opening += (weights[indexOpening]);
                ending += (weights[indexEnding]);
            }
        }
        opening *= color;
        ending *= color;

        const int pieceEval = (WP - BP) * pawnEval;
        const int kingEvalOp = kingOp * (WK - BK);
        const int kingEvalEnd = kingEnd * (WK - BK);


        opening += kingEvalOp;
        opening += pieceEval;
        opening += tpo;
        opening += bala;
        //opening += mobility;

        ending += kingEvalEnd;
        ending += pieceEval;
        ending += tpo;
        ending += bala;
        //ending += mobility;


        int score = (phase * opening + (stage_size - phase) * ending);
        score = div_round(score, (int) stage_size);


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
extern Weights<short> gameWeights;

#endif

#endif //CHECKERENGINEX_WEIGHTS_H
