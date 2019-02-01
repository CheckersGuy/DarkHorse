//
// Created by robin on 7/11/18.
//

#include <BoardFactory.h>
#include <mutex>
#include <atomic>
#include "Training.h"
#include "boost/algorithm/string/predicate.hpp"

namespace Training {
    std::mt19937_64 generator(getSystemTime());

    void TrainingGame::add(Position position) {
        this->positions.emplace_back(position);
    }


    bool TrainingGame::operator==(const Training::TrainingGame &other) {
        return (other.result==result)&&std::equal(positions.begin(),positions.end(),other.positions.begin());
    }

    bool TrainingGame::operator!=(const Training::TrainingGame &other) {
        return !(other.result==result)&&std::equal(positions.begin(),positions.end(),other.positions.begin());
    }

    bool simGame(TrainingGame &one, TrainingGame &two, float threshHold) {
        float counter = 0;
        float size = std::min(one.positions.size(), two.positions.size());
        if (one.result != two.result) {
            return false;
        }
        for (int i = 0; i < size; ++i) {
            if (one.positions[i] == two.positions[i]) {
                counter++;
            }
            if ((counter / size) >= threshHold) {
                return true;
            }

        }
        return false;
    }



    std::istream& operator>>(std::istream& stream,TrainingGame& current){
        Score result;
        stream.read((char *) &result, sizeof(Score));
        current.result = result;
        int length;
        stream.read((char *) &length, sizeof(int));
        for (int i = 0; i < length; ++i) {
            Position pos;
            stream.read((char *) &pos, sizeof(Position));
            current.add(pos);
        }
        return stream;
    }

    std::ostream& operator<<(std::ostream&stream, TrainingGame game){
        stream.write((char *) (&game.result), sizeof(Score));
        int length = game.positions.size();
        stream.write((char *) &length, sizeof(int));
        stream.write((char *) &game.positions[0], sizeof(Position) * length);
        return stream;
    }


    void saveGames(std::vector<TrainingGame> &games, const std::string file) {
        std::ofstream stream(file);
        std::copy(games.begin(),games.end(),std::ostream_iterator<TrainingGame>(stream));
    }

    TrainingPos seekPosition(const std::ifstream& stream, size_t index){

        size_t position=0;
        size_t counter=0;

    }

}