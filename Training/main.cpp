#include <iostream>
#include "Match.h"
#include "Trainer.h"
#include <HyperLog.h>
#include "Generator.h"
#include <ostream>

int main(int argl, const char **argc) {

    initialize();


/*

    std::vector<Position> position;
    initialize();
    setHashSize(21);
    Board board;
    board = Position::getStartPosition();
    Utilities::createNMoveBook(std::back_inserter(position),5,board,-650,650);
    position.emplace_back(Position::getStartPosition());
    Utilities::write_to_binary<Position>(position.begin(),position.end(),"../Training/Positions/train2.pos");
*/



    Generator generator("base", "train2.pos", "test.train");
    generator.set_num_games(1000);
    generator.set_parallelism(15);
    generator.set_time(100);
    generator.start();







/*
    Match engine_match("check", "old", "test2_file");
    engine_match.setTime(100);
    engine_match.setMaxGames(100000);
    engine_match.setNumThreads(1);
    engine_match.setHashSize(21);
    engine_match.set_play_reverse(true);
    engine_match.start();
*/






    /*  std::vector<Training::Position> set;
      std::ifstream stream("../Training/TrainData/output_file", std::ios::binary);
      Training::TrainData data;
      data.ParseFromIstream(&stream);
      std::mt19937_64 generator;
      std::shuffle(data.mutable_positions()->begin(), data.mutable_positions()->end(), generator);
      std::copy(data.mutable_positions()->begin(), data.mutable_positions()->begin() + 3000, std::back_inserter(set));

      data.ParseFromIstream(&stream);
      Utilities::to_binary_data(set.begin(), set.end(), "examples.data");

  */

/*
    std::ifstream stream("test.train");
    if(!stream.good())
        std::cerr<<"Error"<<std::endl;
    std::istream_iterator<Generator::Sample> begin(stream);
    std::istream_iterator<Generator::Sample> end;

    std::for_each(begin,end,[](Generator::Sample test){
       test.first.printPosition();
       std::cout<<"Result: "<<test.second<<std::endl;
    });

*/


/*
      std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
      Trainer trainer("../Training/TrainData/output_file");
      trainer.setLearningRate(16000);
      trainer.setEpochs(1000);
      trainer.setl2Reg(0.000000000000);
      trainer.setCValue(-1e-3);
      trainer.startTune();
      auto loss = trainer.calculateLoss();
      std::cout << "Loss: " << loss << std::endl;
*/












    return 0;
}
