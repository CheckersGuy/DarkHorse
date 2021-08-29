#include <iostream>
#include "Match.h"
#include "Trainer.h"
#include <HyperLog.h>
#include "Generator.h"
#include <ostream>
#include <iterator>
#include "Network.h"
#include <GameLogic.h>
#include <GeneratorZ.h>
#include <thread>
#include <future>
#include <sys/mman.h>
#include <stdlib.h>
#include <unistd.h>


void remove_duplicates(std::string in_File, std::string out_file) {
    int error_count = 0;
    Zobrist::initializeZobrisKeys();
    std::vector<Sample> in_samples;
    Utilities::read_binary<Sample>(std::back_inserter(in_samples), in_File);
    std::vector<Sample> out_samples;
    std::unordered_set<Sample, SampleHasher> hash_table;
    std::cout << "Number of samples before removing duplicates: " << in_samples.size() << std::endl;
    size_t counter = 0;
    for (Sample sample: in_samples) { ;
        //sample.position.printPosition();
        //we have already seen the sample
        if (hash_table.find(sample) != hash_table.end()) {
            continue;
        }
        if ((counter % 1000000) == 0) {
            std::cout << "Progress: " << ((double) counter) / ((double) in_samples.size()) << std::endl;
        }
        counter++;
        hash_table.insert(sample);
        auto num_pieces = Bits::pop_count(sample.position.BP | sample.position.WP);
        uint32_t WP = sample.position.WP & (~sample.position.K);
        uint32_t BP = sample.position.BP & (~sample.position.K);
        uint32_t WK = sample.position.WP & (sample.position.K);
        uint32_t BK = sample.position.BP & (sample.position.K);

        if (num_pieces > 24 || std::abs(sample.position.color) != 1 || num_pieces == 0 ||
            ((WP & BP) != 0) || ((WK & BK) != 0)) {
            error_count++;
            //sample.position.printPosition();
        }else{
            out_samples.emplace_back(sample);
        }
    }
    std::cout << "Number of Errors: " << error_count << std::endl;
    std::cout << "Number of samples after removing duplicates: " << out_samples.size() << std::endl;
    Utilities::write_to_binary<Sample>(out_samples.begin(), out_samples.end(), out_file);

}

int main(int argl, const char **argc) {

    initialize();

    use_classical(true);


/*
    std::vector<Position> positions;
    Board board;
    TT.resize(21);
    board = Position::getStartPosition();
    Utilities::createNMoveBook(std::back_inserter(positions), 6, board, -3200, 3200);
    Utilities::write_to_binary<Position>(positions.begin(), positions.end(),
                                         "/home/robin/DarkHorse/Training/Positions/train3.pos");
    std::cout << "Positions: " << positions.size() << std::endl;

*/



    remove_duplicates("/home/robin/DarkHorse/Training/TrainData/testinggen2",
                      "/home/robin/DarkHorse/Training/TrainData/testinggenremoved2");







/*


    std::vector<Sample> opening;
    std::vector<Sample> ending;
    std::vector<Sample> early_ending;

    std::vector<Sample> data;

    Utilities::read_binary<Sample>(std::back_inserter(data),
                                   "/home/robin/DarkHorse/Training/TrainData/test100removed");

    for (Sample s : data) {
        auto num = Bits::pop_count(s.position.WP | s.position.BP);
        if (num <= 6) {
            ending.emplace_back(s);
        } else if (num > 6 && num <= 12) {
            early_ending.emplace_back(s);
        } else {
            opening.emplace_back(s);
        }
    }
    Utilities::write_to_binary<Sample>(opening.begin(), opening.end(),
                                       "/home/robin/DarkHorse/Training/TrainData/test100open.train");
    Utilities::write_to_binary<Sample>(ending.begin(), ending.end(),
                                       "/home/robin/DarkHorse/Training/TrainData/test100end.train");

    Utilities::write_to_binary<Sample>(opening.begin(), opening.end(),
                                       "/home/robin/DarkHorse/Training/TrainData/test100open.train");
    Utilities::write_to_binary<Sample>(early_ending.begin(), early_ending.end(),
                                       "/home/robin/DarkHorse/Training/TrainData/test100early_end.train");





*/


    Generator generator("test4", "train3.pos", "/home/robin/DarkHorse/Training/TrainData/testinggen2");
    generator.set_num_games(10000000);
    generator.set_hash_size(20);
    generator.set_parallelism(95);
    generator.set_time(50);
    generator.startx();







/*

    Match engine_match("fix8", "fix4");
    engine_match.setTime(100);
    engine_match.setMaxGames(100000);
    engine_match.setNumThreads(14);
    engine_match.setHashSize(21);
    engine_match.start();

*/














/*
    std::vector<Sample> test;
    Utilities::read_binary<Sample>(std::back_inserter(test), "/home/robin/DarkHorse/Training/TrainData/test100removed");
    for (auto i = 0; i < test.size(); ++i) {
        Sample s = test[i];
        auto num_pieces = Bits::pop_count(s.position.BP | s.position.WP);
        if (num_pieces > 24) {
            s.position.printPosition();
            for(auto k=0;k<100;++k){
                test[i-k].position.printPosition();
                std::cout<<std::endl;
            }
            break;
        }
    }
*/
    //0.160792
    std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
    Trainer trainer("/home/robin/DarkHorse/Training/TrainData/testinggenremoved2");
    trainer.setLearningRate(40000);
    trainer.setEpochs(100);
    trainer.setl2Reg(0.000000000000);
    trainer.setCValue(-6e-4);
    trainer.startTune();
    auto loss = trainer.calculateLoss();
    std::cout << "Loss: " << loss << std::endl;


    return 0;
}
