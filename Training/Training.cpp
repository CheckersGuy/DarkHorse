//
// Created by robin on 7/11/18.
//


#include <SMPLock.h>
#include "Training.h"

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
        stream.close();
    }


    void removeDuplicates(std::vector<TrainingGame>& games){
        int counter=0;
        std::unordered_set<TrainingGame,Training::TrainingHash,Training::TrainingComp>mySet;
        std::for_each(games.begin(),games.end(),[&](const TrainingGame& game){
            mySet.insert(game);
        });
        games.clear();
        std::copy(mySet.begin(),mySet.end(),std::back_inserter(games));
    }



}