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
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

void generate_depth_data(int depth, std::string in_data, std::string out_data) {
    //using processes instead of threads
    const int parallelism = 16;

    std::vector<Sample> samples;
    Utilities::read_binary<Sample>(std::back_inserter(samples), in_data);
    const size_t size = samples.size();
    Sample *some_data;
    TrainSample *buffer;
    std::atomic<int> *atomic_counter;
    some_data = (Sample *) mmap(NULL, sizeof(Sample) * size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    buffer = (TrainSample *) mmap(NULL, sizeof(TrainSample) * size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
                                  -1, 0);
    atomic_counter = (std::atomic<int> *) mmap(NULL, sizeof(std::atomic<int>), PROT_READ | PROT_WRITE,
                                               MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    atomic_counter->store(0);

    for (auto i = size_t{0}; i < size; ++i) {
        some_data[i] = samples[i];
    }
    samples = std::vector<Sample>();

    pid_t pid;

    size_t min_chunk_size = size / parallelism;
    size_t remainder = size - min_chunk_size * parallelism;
    std::vector<size_t> chunks(parallelism, min_chunk_size);
    for (auto i = 0; i < remainder; ++i) {
        auto index = remainder % parallelism;
        chunks[index] += 1;
    }

    std::cout << "Test" << std::endl;
    std::cout << "Sum: " << std::reduce(chunks.begin(), chunks.end()) << std::endl;
    std::copy(chunks.begin(), chunks.end(), std::ostream_iterator<size_t>(std::cout, " "));
    std::cout << std::endl;
    size_t start_index = 0u;
    for (auto i = 0; i < parallelism; ++i) {
        pid = fork();
        if (pid < 0) {
            std::cerr << "Error" << std::endl;
            std::exit(-1);
        } else if (pid == 0) {
            initialize();
            use_classical(true);
            TT.resize(20);
            //child process
            std::cout << "Child: " << i << std::endl;
            for (size_t k = start_index; k < start_index + chunks[i]; ++k) {
                Sample s = some_data[k];
                Position pos = s.position;
                Board board;
                board = pos;
                auto eval = board.getMover() * searchValue(board, depth, 1000000, false);
                if (std::abs(eval) >= 9000)
                    eval = 9000 * ((eval >= 0) ? 1 : -1);

                atomic_counter->operator++();
                TrainSample x;
                x.pos = pos;
                x.evaluation = eval;
                x.result = s.result;
                buffer[k] = x;
            }
            std::exit(1);
        }
        start_index += chunks[i];
    }

    if (pid < 0) {
        std::cerr << "Error" << std::endl;
        std::exit(-1);
    }
    std::vector<TrainSample> test_buffer;
    if (pid > 0) {


        while (atomic_counter->load() < size) {
            std::cout << "Counter: " << atomic_counter->load() << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }


        int status;
        for (int j = 0; j < parallelism; ++j) {
            wait(&status);
        }

        for (int j = 0; j < size; ++j) {
            TrainSample s = buffer[j];
            test_buffer.emplace_back(s);
        }
        Utilities::write_to_binary<TrainSample>(test_buffer.begin(), test_buffer.end(), out_data);


        munmap(some_data, size * sizeof(Sample));
        munmap(buffer, size * sizeof(TrainSample));
        std::exit(1);
    }


}


int main(int argl, const char **argc) {

    initialize();
    TT.resize(20);
    use_classical(true);

    generate_depth_data(12, "/home/robin/DarkHorse/Training/TrainData/small_dataset", "eval_data_depth12");

    //creating a subset of the small-dataset consisting of only positions with >6 pieces
/*

    std::vector<TrainSample> samples;
    Utilities::read_binary<TrainSample>(std::back_inserter(samples),
                                        "/home/robin/DarkHorse/cmake-build-debug/eval_data_depth7");

    samples.erase(std::remove_if(samples.begin(), samples.end(), [](TrainSample s) {
        auto num_pieces = Bits::pop_count(s.pos.BP | s.pos.WP);
        return num_pieces >6 ;
    }), samples.end());
    std::cout<<samples.size()<<std::endl;

    Utilities::write_to_binary<TrainSample>(samples.begin(),samples.end(),"endgame_data_depth7");

    std::cout<<"Samples: "<<samples.size()<<std::endl;


    return 0;
*/


/*
 * VERY IMPORTANT I NEED TO REDO SOME OF THE TRAINING STUFF AND DEFINE SOME MAX_EVAL CONSTANT
 * AS THIS BELOW SEEMED TO BE WORKING MUCH MUCH BETTER THAN I HAD EXPECTED
    std::vector<TrainSample> samples;
    Utilities::read_binary<TrainSample>(std::back_inserter(samples),"opening_data");
    for(auto& data : samples){
        data.evaluation = std::clamp(data.evaluation,-9000,9000);
    }
    Utilities::write_to_binary<TrainSample>(samples.begin(),samples.end(),"opening_data2");
*/


/*
    network.load("/home/robin/DarkHorse/cmake-build-debug/nocolor2col.weights");
    network.addLayer(Layer{120, 256});
    network.addLayer(Layer{256, 32});
    network.addLayer(Layer{32, 32});
    network.addLayer(Layer{32, 1});
    network.init();
*/


    /*   Position test;
       test = Position::getStartPosition();
       network.set_input(test);
       test.printPosition();
       std::cout << network.forward_pass() << std::endl;



          std::vector<Sample> positions;


          Utilities::read_binary<Sample>(std::back_inserter(positions),"/home/robin/DarkHorse/Training/TrainData/examples.data");


          for(Sample p : positions){
              p.position.printPosition();
              std::cout<<"Net_eval: "<<network.evaluate(p.position)<<std::endl;
              std::cout<<"Weights_eval: "<<gameWeights.evaluate(p.position,0)<<std::endl;
              std::cout<<std::endl;
              std::cout<<std::endl;
          }
   */








    //playing a simple game using only the eval








/*
    std::vector<Position> openings;

    Utilities::read_binary<Position>(std::back_inserter(openings),
                                     "/home/robin/DarkHorse/Training/Positions/3move.pos");
    Position start = openings[32];
    Board board;
    board = start;
    for (auto i = 0; i < 500; ++i) {
        board.getPosition().printPosition();
        std::cout << "FenString: " << board.getPosition().get_fen_string() << std::endl;
        std::cout << std::endl;
        std::cout << std::endl;
        if (i % 2 == 0)
            use_classical(false);
        else
            use_classical(true);

        Move best;
        searchValue(board, best, 1, 10000000, true);
        board.makeMove(best);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000));

        std::cout << std::endl;


    }
*/










/*



    Board board;

    std::vector<Sample> samples;
    Utilities::read_binary<Sample>(std::back_inserter(samples), "../Training/TrainData/small_dataset");
    std::mt19937_64 generator(213231ull);
    std::shuffle(samples.begin(),samples.end(),generator);
    std::cout << samples.size() << std::endl;

    int counter =0;
    for (Sample s : samples) {

        if(counter>=100)
            break;
        counter++;
        Position pos = s.position;
        if (pos.isEnd())
            continue;

        pos.printPosition();
        std::cout<<"Evaluation: "<<network.evaluate(pos)<<std::endl;
        std::cout << std::endl;
    }
*/




/*

     Generator generator("Generator", "train2.pos", "temp");
     generator.set_num_games(1000000);
     generator.set_hash_size(25);
     generator.set_parallelism(7);
     generator.set_time(100);
     generator.start();
*/
















    /*   Match engine_match("scal3", "moredata3");
       engine_match.setTime(100);
       engine_match.setMaxGames(100000);
       engine_match.setNumThreads(6);
       engine_match.setHashSize(22);
       engine_match.start();
*/








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
