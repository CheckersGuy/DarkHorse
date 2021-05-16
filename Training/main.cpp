#include <iostream>
#include "Match.h"
#include "Trainer.h"
#include <HyperLog.h>
#include "Generator.h"
#include <ostream>
#include <iterator>
#include "Network.h"
int main(int argl, const char **argc) {

    initialize();
    TT.resize(20);
/*


    Network net;

    net.load("test.weights");;
    net.addLayer(Layer{121, 256});
    net.addLayer(Layer{256, 32});
    net.addLayer(Layer{32, 32});
    net.addLayer(Layer{32, 1});

    net.init();

    //checking if net output is correct

    std::vector<Sample> positions;

    Utilities::read_binary<Sample>(std::back_inserter(positions),"/home/robin/DarkHorse/Training/TrainData/examples.data");

    for(Sample p : positions){
        p.position.printPosition();
        net.set_input(p.position);
        std::cout<<"Net_eval: "<<sigmoid(net.forward_pass())<<std::endl;
        std::cout<<std::endl;
        std::cout<<std::endl;
    }

*/



    //playing a simple game using only the eval
/*
    std::vector<Position>openings;

    Utilities::read_binary<Position>(std::back_inserter(openings), "/home/robin/DarkHorse/Training/Positions/train2.pos");
    Position start = Position::getStartPosition();
    Board board;
    board = start;
    for (auto i = 0; i < 500; ++i) {
        board.getPosition().printPosition();
        std::cout << std::endl;
        std::cout << std::endl;
        MoveListe liste;
        getMoves(board.getPosition(), liste);
        Move best;

        int min_value = std::numeric_limits<int>::max();
        for (Move m : liste) {
            Position copy = board.getPosition();
            copy.makeMove(m);
            int temp = (i % 2 == 0) ? net.evaluate(copy) : gameWeights.evaluate(copy, 0);
            temp*=copy.getColor();
            std::cout << temp << std::endl;
            if (temp < min_value) {
                min_value = temp;
                best = m;
            }
        }

        board.makeMove(best);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    }


    std::cout << std::endl;*/






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


   /* Generator generator("Generator", "train2.pos", "temp");
    generator.set_num_games(1000000);
    generator.set_hash_size(25);
    generator.set_parallelism(7);
    generator.set_time(100);
    generator.start();


*/




      /* Match engine_match("network", "dummy");
       engine_match.setTime(500);
       engine_match.setMaxGames(100000);
       engine_match.setNumThreads(4);
       engine_match.setHashSize(23);
       engine_match.start();
*/








    //figuring out a good value for the constant c
    //1. try -2e-3 -> loss: 0.205374
    //2. try -1e-3 -> loss:



  /*   std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
      Trainer trainer("../Training/TrainData/test3.train");
      trainer.setLearningRate(15000);
      trainer.setEpochs(1000);
      trainer.setl2Reg(0.000000000000);
      trainer.setCValue(-1e-3);
      trainer.startTune();
      auto loss = trainer.calculateLoss();
      std::cout << "Loss: " << loss << std::endl;
*/












    return 0;
}
