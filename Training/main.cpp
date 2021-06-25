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
    //rewriting this a little to be using memory backed files
    //using processes instead of threads
    const int parallelism = 16;


    network.load("testbig");
    network.addLayer(Layer{120, 256});
    network.addLayer(Layer{256, 32});
    network.addLayer(Layer{32, 32});
    network.addLayer(Layer{32, 1});

    network.init();
    network2.load("endtest");
    network2.addLayer(Layer{120, 256});
    network2.addLayer(Layer{256, 32});
    network2.addLayer(Layer{32, 32});
    network2.addLayer(Layer{32, 1});

    network2.init();

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
            use_classical(false);
            TT.resize(20);
            //child process
            std::cout << "Child: " << i << std::endl;
            for (size_t k = start_index; k < start_index + chunks[i]; ++k) {
                Sample s = some_data[k];
                Position pos = s.position;
                Board board;
                board = pos;
                int eval = board.getMover() * searchValue(board, depth, 1000000, false);
                eval = std::clamp(eval, -900, 900);

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

struct PosHasher {

    uint64_t operator()(Position pos) const {
        return Zobrist::generateKey(pos);
    }
};

template<typename T>
void remove_duplicates(std::string in_File, std::string out_file) {
    Zobrist::initializeZobrisKeys();
    std::vector<T> in_samples;
    Utilities::read_binary<T>(std::back_inserter(in_samples), in_File);
    std::vector<T> out_samples;
    std::unordered_set<Position, PosHasher> hash_table;
    std::cout << "Number of samples before removing duplicates: " << in_samples.size() << std::endl;
    for (auto sample : in_samples) {
        Position curr;
        if constexpr (std::is_same_v<T, Sample>) {
            curr = sample.position;
        } else {
            curr = sample.pos;
        }
        //we have already seen the sample
        if (hash_table.find(curr) != hash_table.end()) {
            continue;
        }
        hash_table.insert(curr);
        out_samples.emplace_back(sample);
    }
    std::cout << "Number of samples after removing duplicates: " << out_samples.size() << std::endl;
    Utilities::write_to_binary<T>(out_samples.begin(), out_samples.end(), out_file);

}

int main(int argl, const char **argc) {

    initialize();
    //generate_depth_data(12, "/home/robin/DarkHorse/Training/TrainData/openingdataremoved", "/home/robin/DarkHorse/Training/TrainData/depth12dataopen");
    //remove_duplicates<Sample>("/home/robin/DarkHorse/Training/TrainData/openingdata","/home/robin/DarkHorse/Training/TrainData/openingdataremoved");




    //creating a subset of the small-dataset consisting of only positions with >6 pieces

    /* std::vector<Sample> samples;
     Utilities::read_binary<Sample>(std::back_inserter(samples), "/home/robin/DarkHorse/Training/TrainData/openingdata");


     for (auto &sample  : samples) {
         sample.position.printPosition();
         std::cout<<"res: "<<sample.result<<std::endl;
     }*/
    //Utilities::write_to_binary<TrainSample>(samples.begin(),samples.end(),"reinfopendepth9");



/*

    std::vector<TrainSample> positions;
    std::vector<TrainSample> nextPos;
    Utilities::read_binary<TrainSample>(std::back_inserter(positions),
                                        "/home/robin/DarkHorse/Training/TrainData/opening_data_depth9_bigremoved");
    int max =0;
    for (auto &sample : positions) {
        max = std::max(max,sample.evaluation);
        sample.evaluation = std::clamp(sample.evaluation, -6000, 6000);
        nextPos.emplace_back(sample);
    }
    std::cout<<"Maxeval "<<max<<std::endl;
    Utilities::write_to_binary<TrainSample>(nextPos.begin(), nextPos.end(),
                                            "/home/robin/DarkHorse/Training/TrainData/opening_data_depth9_bigremoved2");

*/




/*
    std::vector<Sample> samples;
    Utilities::read_binary<Sample>(std::back_inserter(samples),
                                        "/home/robin/DarkHorse/Training/TrainData/test3.train");

    size_t total = samples.size();
    samples.erase(std::remove_if(samples.begin(), samples.end(), [](Sample s) {
        auto num_pieces = Bits::pop_count(s.position.BP | s.position.WP);
        return num_pieces  <=6 ;
    }), samples.end());
    std::cout<<samples.size()<<std::endl;

    Utilities::write_to_binary<Sample>(samples.begin(),samples.end(),"openingdata");

    std::cout<<"Samples: "<<(double)samples.size()/((double)total)<<std::endl;


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

     Generator generator("Generator", "train2.pos", "temp");
     generator.set_num_games(1000000);
     generator.set_hash_size(25);
     generator.set_parallelism(7);
     generator.set_time(100);
     generator.start();
*/







    Match engine_match("base", "master");
    engine_match.setTime(100);
    engine_match.setMaxGames(100000);
    engine_match.setNumThreads(5);
    engine_match.setHashSize(21);
    engine_match.start();






    /*   network.load("test_open_scalxx9_big.weights");
       network.addLayer(Layer{120, 256});
       network.addLayer(Layer{256, 32});
       network.addLayer(Layer{32, 32});
       network.addLayer(Layer{32, 1});

       network.init();

       network2.load("test_end_scalxx9_big.weights");
       network2.addLayer(Layer{120, 256});
       network2.addLayer(Layer{256, 32});
       network2.addLayer(Layer{32, 32});
       network2.addLayer(Layer{32, 1});

       network2.init();


       Position pos = Position::pos_from_fen("W:W18,19,21,24,25,26,27,28,29,30,31,32:B1,2,3,4,5,7,8,9,11,12,13,15");

       std::vector<Position> positions;
       Utilities::read_binary<Position>(std::back_inserter(positions),"/home/robin/DarkHorse/Training/Positions/train2.pos");



       for(Position p : positions){
           network.init();
           network.set_input(p);
           float test = network.forward_pass();
           float test2 = network.compute_incre_forward_pass(p);
           if(std::abs(test-test2)>=0.01f){
               std::cout<<test<<std::endl;
               std::cout<<test2<<std::endl;
               p.printPosition();
               std::cout<<p.get_fen_string()<<std::endl;

           }
       }

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
