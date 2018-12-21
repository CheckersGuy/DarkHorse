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

namespace Training {


    struct TrainingData;


    struct TrainingPos {
        Position pos;
        Score result;

        TrainingPos() = default;

        TrainingPos(Position pos, Score result) : pos(pos), result(result) {};

        void print();
    };

    struct TrainingGame {
        std::vector<Position> positions;
        Score result = DRAW;

        void extract(TrainingData &data);

        void add(Position position);

        void print();
    };

    struct TrainingData {
        std::vector<TrainingPos> positions;

        TrainingData(TrainingData &data, std::function<bool(TrainingPos)> func);

        TrainingData(const TrainingData &data);

        TrainingData(const std::string file);

        TrainingData() = default;

        std::size_t length();

        void add(TrainingPos pos);

        size_t find(TrainingPos pos);

        void save(const std::string file);

        void shuffle();

        inline TrainingPos operator[](int index) {
            return positions[index];
        }
    };


    void saveGames(std::vector<TrainingGame> &games, const std::string file);


    void loadGames(std::vector<TrainingGame> &games, const std::string file);


    inline double sigmoid(double c, double value) {
        return 1.0 / (1.0 + std::exp(c * value));
    }

    inline double sigmoidDiff(double c, double value) {
        return c * (sigmoid(c, value) * (sigmoid(c, value) - 1.0));
    }



}


#endif //TRAINING_TRAINING_H
