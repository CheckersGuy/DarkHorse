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
constexpr std::array<uint64_t, 8> powers = {1ull, 5ull, 25ull, 125ull, 625ull, 3125ull, 15625ull, 78125ull};
constexpr size_t SIZE = 390625ull * 9ull * 2ull;

namespace fs = std::filesystem;



inline bool is_root_user();


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
    size_t index = 0ull;
    uint32_t pieces = reg & (pos.BP | pos.WP);
    size_t counter = 0ull;
    const uint32_t BP = pos.BP & (~pos.K);
    const uint32_t WP = pos.WP & (~pos.K);
    const uint32_t BK = pos.BP & pos.K;
    const uint32_t WK = pos.WP & pos.K;
    while (pieces) {
        uint32_t lsb = (pieces & ~(pieces - 1u));
        pieces &= pieces - 1u;
        size_t current = ((BP & lsb) != 0u) * 3ull + ((WP & lsb) != 0u) * 4ull + ((BK & lsb) != 0u) * 1ull +
                         ((WK & lsb) != 0u) * 2ull;
        index += powers[counter++] * current;

    }
    return index;
}

template<typename T>
struct Weights {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    T kingOp, kingEnd;
    std::unique_ptr<T[]> weights;

    Weights() : kingOp(50), kingEnd(50), weights(std::make_unique<T[]>(SIZE)) {
        std::fill(weights.get(), weights.get() + SIZE, T{0});
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

    template<typename RunType=uint32_t>
    void loadWeights(const std::string &path) {
        static_assert(std::is_unsigned<RunType>::value);
        fs::path home_path(getenv("HOME"));
        home_path /= "Dokumente";

        auto w_direc = has_directory(home_path, "CWeights");
        if (!w_direc.has_value()) {
            const fs::path tmp_path("/tmp");
            w_direc = has_directory(tmp_path,"CWeights");
        }
        if(!w_direc.has_value()){
            std::cerr<<"Could not find the weights_folder"<<std::endl;
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
            std::cerr<<"Error"<<std::endl;
            return;
        }
        using DataType = double;
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
            auto temp = static_cast<T>(first);
            for (RunType i = 0u; i < length; ++i) {
                weights[counter] = temp;
                counter++;
            }
        }
        DataType kingOpVal, kingEndVal;
        stream.read((char *) &kingOpVal, sizeof(DataType));
        stream.read((char *) &kingEndVal, sizeof(DataType));
        this->kingOp = static_cast<T>(kingOpVal);
        this->kingEnd = static_cast<T>(kingEndVal);
        stream.close();
    }

    template<typename RunType=uint32_t>
    void storeWeights(const std::string &path) {
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
        stream.close();
    }


    Value evaluate(Position pos) const {

        constexpr int pawnEval = 100 * scalfac;
        const Color color = pos.getColor();
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

        for (uint32_t j = 0; j < 3; ++j) {
            for (uint32_t i = 0; i < 3; ++i) {
                const uint32_t curRegion = region << (8u * j + i);
                size_t indexOpening = 18u * getIndex(curRegion, pos) + 3u * j + i;
                size_t indexEnding = 18u * getIndex(curRegion, pos) + 9u + 3u * j + i;
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
