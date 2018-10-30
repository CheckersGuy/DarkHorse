//
// Created by robin on 7/11/18.
//

#include <BoardFactory.h>
#include <mutex>
#include <atomic>
#include "Training.h"
#include "boost/algorithm/string/predicate.hpp"

namespace Training {
    std::mt19937 generator(123123123);

    void TrainingData::shuffle() {
        std::shuffle(positions.begin(), positions.end(), generator);
    }

    void TrainingGame::extract(Training::TrainingData &data) {
        for (Position pos : positions) {
            data.add(TrainingPos(pos, this->result));
        }
    }

    void TrainingGame::print() {
        for (Position pos : positions) {
            pos.printPosition();
            std::cout << "##########################" << std::endl;
        }
        std::cout << std::endl;
    }

    void TrainingGame::add(Position position) {
        this->positions.emplace_back(position);
    }

    void TrainingPos::print() {
        pos.printPosition();
    }

    void TrainingData::add(Training::TrainingPos pos) {
        positions.emplace_back(pos);
    }

    TrainingData::TrainingData(TrainingData &data, std::function<bool(TrainingPos)> func) {
        for (TrainingPos pos : data.positions) {
            if (func(pos)) {
                add(pos);
            }
        }
    }

    TrainingData::TrainingData(const TrainingData &data) {
        for (TrainingPos position : data.positions) {
            add(position);
        }
    }

    TrainingData::TrainingData(const std::string file) {
        std::ifstream stream(file, std::ios::binary);
        if (!stream.good())
            exit(EXIT_FAILURE);

        while (!stream.eof()) {
            TrainingPos current;
            stream.read((char *) &current, sizeof(TrainingPos));
            add(current);
        }
        stream.close();
    }

    int TrainingData::find(TrainingPos pos) {
        int counter = 0;
        for (TrainingPos current : positions) {
            if (current.pos == pos.pos && current.result == pos.result) {
                counter++;
            }
        }
        return counter;
    }

    void TrainingData::save(const std::string file) {
        std::string path = "TrainData/" + file;
        std::ofstream stream(path, std::ios::binary | std::ios::app);

        if (!stream.good()) {
            return exit(EXIT_FAILURE);
        }
        stream.write((char *) (&positions[0]), sizeof(TrainingPos) * positions.size());
        stream.close();

    }


    int TrainingData::length() {
        return positions.size();
    }

    void saveGames(std::vector<TrainingGame> &games, const std::string file) {
        std::ofstream stream(file,  std::ios::binary);
        if (!stream.good()) {
            throw std::string("could not find the file");
        }
        for (TrainingGame &game : games) {
            stream.write((char *) (&game.result), sizeof(Score));
            int length = game.positions.size();
            stream.write((char *) &length, sizeof(int));
            stream.write((char *) &game.positions[0], sizeof(Position) * length);
        }
        stream.close();

    }

    void loadGames(std::vector<TrainingGame> &games, const std::string file) {
        std::ifstream stream(file, std::ios::binary);

        if (!boost::iends_with(file, ".game")) {
            throw std::string("isn't a game file");
        }

        if (!stream.good()) {
            std::cerr << "Error" << std::endl;
            throw std::string("Could not find the file");
        }


        while (!stream.eof()) {
            TrainingGame current;
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
            games.emplace_back(current);
        }
        stream.close();
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


    Value  quiescene(Weights<double>&weights, Board &board, Value alpha, Value beta, int ply, uint64_t endTime) {

        if (ply >= MAX_PLY) {
            return board.getMover() * weights.evaluate(*board.getPosition());
        }

        MoveListe moves;
        getCaptures(board, moves);
        nodeCounter++;
        Value bestValue= -INFINITE;

        if (moves.length() == 0) {
            Line localPV;
            if(board.getPosition()->hasThreat()){
                return alphaBeta(board,alpha,beta,localPV,alpha!=beta-1,ply+1,1,false);
            }

            if(board.getPosition()->isWipe()){
                return Value::loss(board.getMover(), ply);
            }

            bestValue =board.getMover()*weights.evaluate(*board.getPosition());
            if(bestValue>=beta){
                return bestValue;
            }
        }

        for (int i = 0; i < moves.length(); ++i) {
            board.makeMove(moves.liste[i]);
            Value value = ~quiescene(weights,board, ~beta, ~alpha, ply + 1,endTime);
            board.undoMove();

            if (value > bestValue) {
                bestValue = value;
                if (value >= beta) {
                    break;
                }
                if(value>alpha){
                    alpha=value;
                }

            }
        }
        return bestValue;
    }





}