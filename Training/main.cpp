#include <iostream>
#include "Match.h"
#include "Trainer.h"
#include "Generator.h"
#include <ostream>
#include <iterator>
#include "Network.h"
#include <GameLogic.h>
#include <thread>
#include <future>
#include <sys/mman.h>
#include <stdlib.h>
#include <unistd.h>
#include <SampleFilter.h>
#include <SampleFilter2.h>

void remove_duplicates(std::string in_File, std::string out_file) {
    int error_count = 0;
    Zobrist::initializeZobrisKeys();
    std::vector<Sample> in_samples;
    Utilities::read_binary<Sample>(std::back_inserter(in_samples), in_File);
    std::vector<Sample> out_samples;
    std::unordered_set<Sample, std::hash<Sample>> hash_table;
    std::cout << "Number of samples before removing duplicates: " << in_samples.size() << std::endl;
    size_t counter = 0;

    auto t1 = std::chrono::high_resolution_clock::now();

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
        } else {
            out_samples.emplace_back(sample);
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    auto dist = t2-t1;
    std::cout<<"It took: "<<dist.count()/1000000<<std::endl;

    std::cout << "Number of Errors: " << error_count << std::endl;
    std::cout << "Number of samples after removing duplicates: " << out_samples.size() << std::endl;
   // Utilities::write_to_binary<Sample>(out_samples.begin(), out_samples.end(), out_file);

}

int main(int argl, const char **argc) {

    initialize();

    use_classical(true);


/*

    std::vector<Position> positions;
    Board board;
    TT.resize(21);
    board = Position::getStartPosition();
    Utilities::createNMoveBook(std::back_inserter(positions), 6, board, -2000, 2000);
    Utilities::write_to_binary<Position>(positions.begin(), positions.end(),
                                         "/home/robin/DarkHorse/Training/Positions/train2.pos");
    std::cout << "Positions: " << positions.size() << std::endl;


*/





//number of samples: 200117





/*
    remove_duplicates("/home/robin/DarkHorse/Training/TrainData/bloom.check",
                      "/home/robin/DarkHorse/Training/TrainData/dummyremovethis2");


    std::vector<Sample> removed;
    std::ifstream stream("../Training/TrainData/bloom.check");
    std::istream_iterator<Sample> begin(stream);
    std::istream_iterator<Sample> end;

    auto t1 = std::chrono::high_resolution_clock::now();

    SampleFilter2 filter(14377587567, 10);

    auto count = std::count_if(begin, end, [&](Sample s) {
        if(filter.has(s)){
            return false;
        }
        filter.insert(s);
        return true;
    });
    std::cout<<"NumElements: "<<count<<std::endl;

    auto t2 = std::chrono::high_resolution_clock::now();
    auto dist = t2-t1;
    std::cout<<"It took: "<<dist.count()/1000000<<std::endl;

    std::cout<<removed.size()<<std::endl;

    return 0;


*/



    Generator generator("train.pos", "/home/robin/DarkHorse/Training/TrainData/bloom");
    generator.set_hash_size(20);
    generator.set_buffer_clear_count(150000);
    generator.set_parallelism(95);
    generator.set_time(50);
    generator.startx();
    
/*

    Match engine_match("bloom5", "bloom4");
    engine_match.setTime(100);
    engine_match.setMaxGames(100000);
    engine_match.setNumThreads(14);
    engine_match.setHashSize(20);
    engine_match.start();
*/



    //0.193772
    std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
    Trainer trainer("/home/robin/DarkHorse/Training/TrainData/bloom");
    trainer.setLearningRate(140000);
    trainer.setEpochs(100);
    trainer.setl2Reg(0.000000000000);
    trainer.setCValue(-7e-4);
    trainer.startTune();
    auto loss = trainer.calculateLoss();
    std::cout << "Loss: " << loss << std::endl;


    return 0;
}
