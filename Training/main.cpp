#include <iostream>
#include "Match.h"
#include "Trainer.h"
#include "proto/Training.pb.h"

struct HelpInserter {
    Training::TrainData &data;
    using value_type = Position;

    void push_back(const Position pos) {
        auto p = data.add_positions();
        p->set_mover((pos.color == BLACK) ? Training::BLACK : Training::WHITE);
        p->set_wp(pos.WP);
        p->set_bp(pos.BP);
        p->set_k(pos.K);
    }

};


int main() {
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






    Match engine_match("generator", "generator", "output_file2");
    engine_match.setTime(100);
    engine_match.setMaxGames(1);
    engine_match.setNumThreads(1);
    engine_match.setHashSize(20);
    engine_match.set_play_reverse(false);
    engine_match.start();









/*
    Match engine_match("new_light4", "new_light3", "match_file");
    engine_match.setTime(100);
    engine_match.setMaxGames(20000);
    engine_match.setNumThreads(15);
    engine_match.setHashSize(22);
    engine_match.set_play_reverse(true);
    engine_match.start();
*/





    //0.187126
    //0.186267
    //0.18615
    //0.18618
    //0.186162
    //0.185978
    //0.185965

/*
    std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
    Trainer trainer("output_file");
    trainer.setLearningRate(500000);
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
