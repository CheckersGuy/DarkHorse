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

        auto begin(){
            return positions.begin();
        }

        auto end(){
            return positions.end();
        }
    };

    struct TrainingHash{
    public:

        TrainingHash()=default;

        uint64_t operator()(TrainingGame& game){
            uint64_t key=0;
            int length= std::min(game.positions.size(),static_cast<size_t>(40));
            for(int i=0;i<length;++i){
                key^=Zobrist::generateKey(game.positions[i],game.positions[i].getColor());
            }
        }
    };

    struct TrainingComp{

        bool operator()(TrainingGame& one, TrainingGame& two){
            int length=one.positions.size();
            if(two.positions.size()<length)
                length=two.positions.size();
            if(40<length)
                length=40;

            return std::equal(one.begin(),one.begin()+length,two.begin());
        }
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
