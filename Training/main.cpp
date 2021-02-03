#include <iostream>
#include "Match.h"
#include "Trainer.h"
#include <HyperLog.h>
#include "Generator.h"

int main(int argl, const char **argc) {
    initialize();
    /*   Training::TrainData data;
       HelpInserter inserter{data};
       Board board;
       board = Position::getStartPosition();
       Utilities::createNMoveBook(std::back_inserter(inserter),4,board,-30*scalfac,30*scalfac);
       inserter.push_back(Position::getStartPosition());


       std::cout<<"Test: "<<data.positions_size()<<std::endl;

       std::ofstream stream("genBook2.book");
       data.SerializeToOstream(&stream);
       stream.close();*/
/*

    std::cout << "Starting Match" << std::endl;
    std::cout << "Parallelism: " << std::endl;
    int threads;
    std::cin >> threads;
    std::cout << "MaxGames: " << std::endl;
    int max_games;
    std::cin >> max_games;

    Match engine_match("Generator", "Generator", "../Training/TrainData/output_file");
    engine_match.setTime(100);
    engine_match.setMaxGames(max_games);
    engine_match.setNumThreads(threads);
    engine_match.setHashSize(21);
    engine_match.set_play_reverse(false);
    engine_match.start();


*/


/*

    Generator generator("base", "3move.pos", "test.train");
    generator.set_num_games(100000);
    generator.set_parallelism(16);
    generator.start();
*/




/*

    Match engine_match("base", "base", "test2_file");
    engine_match.setTime(20);
    engine_match.setMaxGames(100000);
    engine_match.setNumThreads(16);
    engine_match.setHashSize(21);
    engine_match.set_play_reverse(false);
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






      std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
      Trainer trainer("../Training/TrainData/output_file");
      trainer.setLearningRate(16000);
      trainer.setEpochs(1000);
      trainer.setl2Reg(0.000000000000);
      trainer.setCValue(-1e-3);
      trainer.startTune();
      auto loss = trainer.calculateLoss();
      std::cout << "Loss: " << loss << std::endl;












    return 0;
}
