#include <iostream>
#include "Match.h"
#include "Trainer.h"
#include <HyperLog.h>
#include "Generator.h"
#include <ostream>
#include <iterator>

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
    generator.set_num_games(1000000);
    generator.set_hash_size(23);
    generator.set_parallelism(7);
    generator.set_time(100);
    generator.start();



/*
    std::vector<Sample> data;
    Utilities::read_binary<Sample>(std::back_inserter(data), "../Training/TrainData/examples.data");

    std::for_each(data.begin(),data.end(),[](Sample&sample){
        Color color=sample.position.getColor();
        Position flipped=sample.position;
        if(sample.position.getColor()==BLACK)
            flipped = sample.position.getColorFlip();
        sample.position = flipped;
        sample.result = color*sample.result;
    });
    Utilities::write_to_binary<Sample>(data.begin(),data.end(),"../Training/TrainData/flipped-examples.train");
*/
/*

       Match engine_match("moredata2", "moredata2");
       engine_match.setTime(100);
       engine_match.setMaxGames(100000);
       engine_match.setNumThreads(6);
       engine_match.setHashSize(23);
       engine_match.start();



*/







    //figuring out a good value for the constant c
    //1. try -2e-3 -> loss: 0.205374
    //2. try -1e-3 -> loss:



/*      std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
      Trainer trainer("../Training/TrainData/test2.train");
      trainer.setLearningRate(15000);
      trainer.setEpochs(1000);
      trainer.setl2Reg(0.000000000000);
      trainer.setCValue(-1e-3);
      trainer.startTune();
      auto loss = trainer.calculateLoss();
      std::cout << "Loss: " << loss << std::endl;*/













    return 0;
}
