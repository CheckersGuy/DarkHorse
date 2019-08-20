//
// Created by robin on 7/11/18.
//

#ifndef TRAINING_TRAINING_H
#define TRAINING_TRAINING_H

#include <unordered_set>
#include "Position.h"
#include "types.h"
#include <vector>
#include <fstream>
#include "Engine.h"
#include <BoardFactory.h>
#include <MGenerator.h>
#include "Weights.h"
#include <algorithm>
#include <functional>
#include <iterator>


namespace Training {

    extern std::mt19937_64 generator;

    struct TrainingPos {
        Position pos;
        Score result;

        TrainingPos() = default;

        TrainingPos(Position pos, Score result) : pos(pos), result(result) {};

    };


    struct TrainingGame {
        std::vector<Position> positions;
        Score result = DRAW;

        void add(Position position);

        bool operator==(const TrainingGame &other);

        bool operator!=(const TrainingGame &other);

        auto begin() {
            return positions.begin();
        }

        auto end() {
            return positions.end();
        }

        auto begin() const {
            return positions.cbegin();
        }

        auto end() const {
            return positions.cend();
        }

        size_t length() const {
            return positions.size();
        }

        void print() {
            std::for_each(begin(), end(), [](Position pos) {
                pos.printPosition();
                std::cout << std::endl;
            });
        }
    };

    inline size_t minValue(size_t val1, size_t val2) {
        return (val1 <= val2) ? val1 : val2;
    }

    struct TrainingHash {
        uint64_t operator()(const TrainingGame &game) const {
            uint64_t key = 0;
            size_t finalLength = minValue(game.length(), 25u);

            for (size_t i = 0; i < finalLength; ++i) {
                key ^= Zobrist::generateKey(game.positions[i], game.positions[i].getColor());
            }
            return key;
        }
    };

    struct TrainingComp {

        bool operator()(const TrainingGame &one, const TrainingGame &two) const {
            if (one.result != two.result)
                return false;


            size_t finalLength = minValue(minValue(one.length(), two.length()), 25u);
            return std::equal(one.begin(), one.begin() + finalLength, two.begin());
        }
    };

    std::istream &operator>>(std::istream &stream, TrainingGame &game);

    std::ostream &operator<<(std::ostream &stream, TrainingGame game);


    void saveGames(std::vector<TrainingGame> &games, const std::string file);


    template<class T=TrainingGame>
    void loadGames(std::vector<T> &games, const std::string file, int limit = std::numeric_limits<int>::max()) {
        static_assert(std::is_same<T, TrainingGame>::value || std::is_same<T, TrainingPos>::value);
        std::ifstream stream(file, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Error" << std::endl;
            throw std::string("Could not find the file");
        }
        int counter = 0;
        while (counter < limit && !stream.eof()) {
            TrainingGame current;
            stream >> current;
            if constexpr(std::is_same<T, TrainingGame>::value) {
                games.emplace_back(current);
            } else if constexpr(std::is_same<T, TrainingPos>::value) {
                std::for_each(current.positions.begin(), current.positions.end(),
                              [&games, &current](Position pos) { games.push_back(TrainingPos(pos, current.result)); });
            }
            counter++;
        }
        stream.close();
    }

    inline double sigmoid(double c, double value) {
       double sig= 1.0/(1.0+std::exp(c*value));

        sig=std::min(sig,0.999999);
        sig=std::max(sig,0.000001);
        return sig;
    }

    inline double sigmoidDiff(double c, double value) {
        return c * (sigmoid(c, value) * (sigmoid(c, value) - 1.0));

    }



    inline double signum(double value){
        if( value==0.0)
            return 0.0;

        return (value>=0)?1.0:-1.0;
    }






    std::vector<TrainingGame> removeDuplicates(std::vector<TrainingGame> &games);


}


#endif //TRAINING_TRAINING_H
