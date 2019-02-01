//
// Created by robin on 7/11/18.
//

#ifndef TRAINING_TRAINING_H
#define TRAINING_TRAINING_H

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
        bool operator==(const TrainingGame& other);
        bool operator!=(const TrainingGame& other);
    };

    std::istream &operator>>(std::istream &stream, TrainingGame &game);

    std::ostream &operator<<(std::ostream &stream, TrainingGame game);


    void saveGames(std::vector<TrainingGame> &games, const std::string file);


    template<class T=TrainingGame> void loadGames(std::vector<T> &games, const std::string file) {
        static_assert(std::is_same<T,TrainingGame>::value||std::is_same<T,TrainingPos>::value);
        std::ifstream stream(file, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Error" << std::endl;
            throw std::string("Could not find the file");
        }
        while (!stream.eof()) {
            TrainingGame current;
            stream >> current;
            if constexpr(std::is_same<T,TrainingGame>::value){
                games.emplace_back(current);
            }else if constexpr(std::is_same<T,TrainingPos>::value){
                std::for_each(current.positions.begin(),current.positions.end(),[&games,&current](Position pos){games.push_back(TrainingPos(pos,current.result));});
            }
        }
        stream.close();
    }

    TrainingPos seekPosition(const std::ifstream& stream,size_t index);


    inline double sigmoid(double c, double value) {
        return 1.0 / (1.0 + std::exp(c * value));
    }

    inline double sigmoidDiff(double c, double value) {
        return c * (sigmoid(c, value) * (sigmoid(c, value) - 1.0));
    }


}


#endif //TRAINING_TRAINING_H
