#include <iostream>
#include "Match.h"
#include "Trainer.h"
#include <HyperLog.h>

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


struct Hasher{

    uint64_t xorshift(const uint64_t& n,int i){
        return n^(n>>i);
    }

    uint64_t operator()(uint64_t n){
        uint64_t p = 0x5555555555555555ull; // pattern of alternating 0 and 1
        uint64_t c = 17316035218449499591ull;// random uneven integer constant;
        return c*xorshift(p*xorshift(n,32),32);
    }
};


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
    engine_match.setHashSize(20);
    engine_match.set_play_reverse(false);
    engine_match.start();

*/






    Match engine_match("t5", "old", "test_file");
    engine_match.setTime(100);
    engine_match.setMaxGames(20000);
    engine_match.setNumThreads(6);
    engine_match.setHashSize(21);
    engine_match.set_play_reverse(true);
    engine_match.start();




/*
    size_t unique_counter =0;

    std::mt19937_64 generator(637231u);
    std::uniform_int_distribution<uint64_t> distrib(0,1000000000ull);

    auto t1 = std::chrono::high_resolution_clock::now();

    HyperLog< uint64_t,4,Hasher> log;
    for (int i = 1; i <= 1000080005ull; ++i) {
        auto value = distrib(generator);
        log.insert(value);
    }
    auto t2 = std::chrono::high_resolution_clock::now();
    auto duration =t2-t1;
    std::cout<<"Unique_Counter: "<<unique_counter<<std::endl;
    std::cout << "Count: " << log.get_count() << std::endl;
    std::cout<<"Time: "<<duration.count()/1000000<<std::endl;
*/



/*

       std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
       Trainer trainer("../Training/TrainData/output_file");
       trainer.setLearningRate(10000);
       trainer.setEpochs(100);
       trainer.setl2Reg(0.000000000000);
       trainer.setCValue(-5e-3);
       trainer.startTune();
       //0.197482
       auto loss = trainer.calculateLoss();
       std::cout << "Loss: " << loss << std::endl;

*/





    return 0;
}
