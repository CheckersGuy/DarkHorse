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


size_t count_unique_elements(std::string input) {
    std::ifstream stream(input, std::ios::binary);
    std::istream_iterator<Sample> begin(stream);
    std::istream_iterator<Sample> end;

    size_t counter = 0;
    size_t total_elements = 0;
    SampleFilter filter(5751035027, 10);

    std::for_each(begin, end, [&](Sample s) {
        if (!filter.has(s)) {
            counter++;
            filter.insert(s);
        }
        total_elements++;
    });

    std::cout << "TotalElements: " << total_elements << std::endl;
    std::cout << "Unique Elements: " << counter << std::endl;

    return counter;
}

void remove_duplicates(std::string input, std::string output) {
    std::ifstream stream(input, std::ios::binary);
    std::istream_iterator<Sample> begin(stream);
    std::istream_iterator<Sample> end;

    size_t counter = 0;
    size_t total_elements = 0;
    SampleFilter filter(5751035027, 10);
    std::vector<Sample> elements;
    std::for_each(begin, end, [&](Sample s) {
        if (!filter.has(s)) {
            counter++;
            filter.insert(s);
            elements.emplace_back(s);
        }
        total_elements++;
    });

    std::ofstream out_stream(output, std::ios::binary);
    std::copy(elements.begin(), elements.end(), std::ostream_iterator<Sample>(out_stream));
    std::cout << "Size after removing: " << elements.size() << std::endl;
    std::cout << "Removed a total of " << total_elements - elements.size() << " elements" << std::endl;
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


    remove_duplicates("/home/robin/DarkHorse/Training/TrainData/bloomcloudrescored",
                      "/home/robin/DarkHorse/Training/TrainData/bloomcloudrescored");

    return 0;

*/







      Utilities::create_samples_from_games("/home/robin/DarkHorse/Training/TrainData/check.games", "/home/robin/DarkHorse/Training/TrainData/check.samples");
      return 0;




 /*   Generator generator("train3.pos", "/home/robin/DarkHorse/Training/TrainData/reinf.games");
    generator.set_hash_size(20);
    generator.set_buffer_clear_count(10000);
    generator.set_parallelism(14);
    generator.set_time(50);
    generator.startx();


*/




    Match engine_match("fun4rescored", "network256x32");
    engine_match.setTime(100);
    engine_match.setMaxGames(100000);
    engine_match.setNumThreads(14);
    engine_match.setHashSize(21);
    engine_match.start();





    // 0.190537  1e-4
    //0.188262   6e-4
    //0.188813 1e-3
    //0.127496
    std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
    Trainer trainer("/home/robin/DarkHorse/Training/TrainData/check.samples");
    trainer.setLearningRate(150000);
    trainer.setEpochs(100);
    trainer.setl2Reg(0.000000000000);
    trainer.setCValue(-1e-3);
    trainer.startTune();
    auto loss = trainer.calculateLoss();
    std::cout << "Loss: " << loss << std::endl;


    return 0;
}
