#include <iostream>
#include "Match.h"
#include "Trainer.h"


int main() {


    initialize();


/*
    Training::TrainData data;
    std::ifstream stream("output_file");
    data.ParseFromIstream(&stream);
    stream.close();
    std::for_each(data.mutable_positions()->begin(), data.mutable_positions()->end(), [](Training::Position &pos) {
        Board board;
        board.getPosition().BP = pos.bp();
        board.getPosition().WP = pos.wp();
        board.getPosition().K = pos.k();
        board.printBoard();
        std::cout << std::endl;
    });
*/



/*

    Match engine_match("generator", "generator", "output_file");
    engine_match.setTime(100);
    engine_match.setMaxGames(2000);
    engine_match.setNumThreads(15);
    engine_match.setHashSize(20);
    engine_match.set_play_reverse(false);
    engine_match.start();


*/

//Matchmaking




    Match engine_match("new_light3", "new_light2", "match_file");
    engine_match.setTime(100);
    engine_match.setMaxGames(2000);
    engine_match.setNumThreads(15);
    engine_match.setHashSize(22);
    engine_match.set_play_reverse(true);
    engine_match.start();

    








    //some loss-values during training
    //0.18219
    //0.181486
    //0.18139








/*
    std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
    Trainer trainer("output_file");
    trainer.setLearningRate(180000);
    trainer.setEpochs(100);
    trainer.setl2Reg(0.000000000000);
    trainer.setCValue(-5e-4);
    trainer.startTune();
    //0.197482
    auto loss = trainer.calculateLoss();
    std::cout << "Loss: " << loss << std::endl;
*/




    return 0;
}
