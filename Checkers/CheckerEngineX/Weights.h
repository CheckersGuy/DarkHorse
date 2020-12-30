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

inline size_t getTopBotIndex(uint32_t reg, const Position &pos) {
    //we can save some memory by using the fact that
    //we can not black/white-man on the promotion squares

    //first we compute the index as we did before for the 6th squares
    //and then we compute the other ones

    size_t index = 0u;

    const uint32_t promo_ring = PROMO_SQUARES_BLACK | PROMO_SQUARES_WHITE;

    const uint32_t BP = pos.BP & (~pos.K);
    const uint32_t WP = pos.WP & (~pos.K);
    const uint32_t BK = pos.BP & pos.K;
    const uint32_t WK = pos.WP & pos.K;
    uint32_t pieces = reg & (pos.BP | pos.WP) & (~promo_ring);

    while (pieces) {
        uint32_t lsb = (pieces & ~(pieces - 1u));
        pieces &= pieces - 1u;
        size_t current = ((BP & lsb) != 0u) * 3ull + ((WP & lsb) != 0u) * 4ull + ((BK & lsb) != 0u) * 1ull +
                         ((WK & lsb) != 0u) * 2ull;
        index += current;
        index = index * 5ull;
    }
    //
    constexpr size_t NUM_PIECES = 0;
    //to be continued


}

inline size_t getIndex(uint32_t reg, const Position &pos) {
    //will return the index for a given position
    size_t index = 0ull;
    uint32_t pieces = reg & (pos.BP | pos.WP);
    const uint32_t BP = pos.BP & (~pos.K);
    const uint32_t WP = pos.WP & (~pos.K);
    const uint32_t BK = pos.BP & pos.K;
    const uint32_t WK = pos.WP & pos.K;


    uint32_t last_piece = (pieces & ~(pieces - 1u));
    pieces ^= last_piece;


    while (pieces) {
        uint32_t lsb = (pieces & ~(pieces - 1u));
        pieces &= pieces - 1u;
        size_t current = ((BP & lsb) != 0u) * 3ull + ((WP & lsb) != 0u) * 4ull + ((BK & lsb) != 0u) * 1ull +
                         ((WK & lsb) != 0u) * 2ull;
        index += current;
        index = index * 5ull;
    }
    index += ((BP & last_piece) != 0u) * 3ull + ((WP & last_piece) != 0u) * 4ull + ((BK & last_piece) != 0u) * 1ull +
             ((WK & last_piece) != 0u) * 2ull;

    return index;
}

template<typename T>
struct Weights {
    static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
    T kingOp, kingEnd;
    std::unique_ptr<T[]> weights;
    std::array<std::array<T, 16>, 7> tempo_ranks;
    T kingMobD, kingMobS;

    Weights() : kingOp(150), kingEnd(150), kingMobD(10), kingMobS(10), weights(std::make_unique<T[]>(SIZE)) {
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


    template<Color color>
    int tempi(Position &pos) const {
        const uint32_t man = pos.getCurrent<color>() & (~pos.K);

        if constexpr(color == BLACK) {
            auto tempi = tempo_ranks[0][(man) & temp_mask];
            tempi += tempo_ranks[1][(man >> 4u) & temp_mask];
            tempi += tempo_ranks[2][(man >> 8u) & temp_mask];
            tempi += tempo_ranks[3][(man >> 12u) & temp_mask];
            tempi += tempo_ranks[4][(man >> 16u) & temp_mask];
            tempi += tempo_ranks[5][(man >> 20u) & temp_mask];
            tempi += tempo_ranks[6][(man >> 24u) & temp_mask];
            return tempi;
        } else {
            auto tempi = tempo_ranks[0][(man >> 28u) & temp_mask];
            tempi += tempo_ranks[1][(man >> 24u) & temp_mask];
            tempi += tempo_ranks[2][(man >> 20u) & temp_mask];
            tempi += tempo_ranks[3][(man >> 16u) & temp_mask];
            tempi += tempo_ranks[4][(man >> 12u) & temp_mask];
            tempi += tempo_ranks[5][(man >> 8u) & temp_mask];
            tempi += tempo_ranks[6][(man >> 4u) & temp_mask];
            return tempi;
        }
    }

    int king_mobiility(Position &pos) const {
        //first for white pieces
        uint32_t kings_white = pos.WP & (~pos.K);
        uint32_t kings_black = pos.BP & (~pos.K);
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

        return count_denied * kingMobD + count_safe * kingMobS;

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
        for (auto i = 0; i < tempo_ranks.size(); ++i) {
            for (auto j = 0; j < 16; ++j) {
                double temp;
                stream.read((char *) &temp, sizeof(DataType));
                tempo_ranks[i][j] = temp;
            }
        }
        DataType mob1, mob2;
        stream.read((char *) &mob1, sizeof(DataType));
        kingMobS = mob1;
        stream.read((char *) &mob2, sizeof(DataType));
        kingMobD = mob2;
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
        stream.write((char *) &kingMobS, sizeof(DataType));
        stream.write((char *) &kingMobD, sizeof(DataType));
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


        int tpo = tempi<WHITE>(pos) - tempi<BLACK>(pos);
        int mobility = king_mobiility(pos);

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
        opening += mobility;

        ending += kingEvalEnd;
        ending += pieceEval;
        ending += tpo;
        ending += mobility;


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
