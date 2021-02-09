#include <iostream>
#include "Match.h"
#include "Trainer.h"
#include <HyperLog.h>
#include "Generator.h"
#include <ostream>

int main(int argl, const char **argc) {

    initialize();

    /*  Board board;

      std::vector<Sample> samples;
      Utilities::read_binary<Sample>(std::back_inserter(samples),"../Training/TrainData/test.train");
      std::cout<<samples.size()<<std::endl;

      for(Sample s : samples){
          Position pos = s.position;
          if(pos.isEnd())
              continue;
          Position temp = Position::pos_from_fen(pos.get_fen_string());
          if(pos!=temp){
              std::cerr<<"Error"<<std::endl;
              std::cerr<<pos.get_fen_string()<<std::endl;
              pos.printPosition();
              std::cout<<std::endl;
              temp.printPosition();
              std::exit(-1);
          }
      }
  */

     Generator generator("Generator", "train2.pos", "temp");
     generator.set_num_games(20000);
     generator.set_parallelism(7);
     generator.set_time(100);
     generator.start();






/*
    Match engine_match("weird2", "ultron");
    engine_match.setTime(100);
    engine_match.setMaxGames(100000);
    engine_match.setNumThreads(7);
    engine_match.setHashSize(21);
    engine_match.start();*/












/*

      std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
      Trainer trainer("../Training/TrainData/test.train");
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
