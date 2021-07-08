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

void generate_depth_data(int depth, std::string in_data, std::string out_data) {
    //rewriting this a little to be using memory backed files
    //using processes instead of threads

/*    network.load("depth3testreinf");
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

    network2.init();*/

    const int parallelism = 16;
    use_classical(false);

    std::vector<Sample> samples;
    Utilities::read_binary<Sample>(std::back_inserter(samples), in_data);
    const size_t size = samples.size();
    Sample *some_data;
    PolySample1 *buffer;
    std::atomic<int> *atomic_counter;


    some_data = (Sample *) mmap(NULL, sizeof(Sample) * size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    buffer = (PolySample1 *) mmap(NULL, sizeof(PolySample1) * size, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
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
            use_classical(true);
            TT.resize(20);
            //child process
            std::cout << "Child: " << i << std::endl;
            for (size_t k = start_index; k < start_index + chunks[i]; ++k) {
                Sample s = some_data[k];
                Position pos = s.position;
                Color color = pos.color;
                Board board;
                board = pos;
                Move best;
                int eval = board.getMover() * searchValue(board, best, depth, 1000000, false);
                eval = std::clamp(eval, -9000, 9000);

                uint32_t mover = best.from;
                Position temp = pos;

                if (color == BLACK) {
                    temp = temp.getColorFlip();
                    mover = getMirrored(mover);
                }

                /*     temp.printPosition();
                     Position test;
                     test.BP = mover;
                     test.printPosition();
                     std::cout << std::endl;
                     std::cout << std::endl;
                     std::cout << std::endl;*/

                atomic_counter->operator++();
                PolySample1 x;
                x.pos = pos;
                x.evaluation = eval;
                x.result = s.result;
                //should be stored from whites perspective
                x.piece_moved = (int) Bits::bitscan_foward(mover);
                buffer[k] = x;
            }
            std::exit(1);
        }
        start_index += chunks[i];
    }
    std::vector<PolySample1> test_buffer;
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
            PolySample1 s = buffer[j];
            test_buffer.emplace_back(s);
        }
        Utilities::write_to_binary<PolySample1>(test_buffer.begin(), test_buffer.end(), out_data);


        munmap(some_data, size * sizeof(Sample));
        munmap(buffer, size * sizeof(PolySample1));
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
    std::unordered_set<Sample, SampleHasher> hash_table;
    std::cout << "Number of samples before removing duplicates: " << in_samples.size() << std::endl;
    for (auto sample : in_samples) {
        Position curr;
        //we have already seen the sample
        if (hash_table.find(sample) != hash_table.end()) {
            continue;
        }
        hash_table.insert(sample);
        out_samples.emplace_back(sample);
    }
    std::cout << "Number of samples after removing duplicates: " << out_samples.size() << std::endl;
    Utilities::write_to_binary<T>(out_samples.begin(), out_samples.end(), out_file);

}

void testing() {
    pthread_mutex_t *pmutex = NULL;
    pthread_mutexattr_t attrmutex;
/* Allocate memory to pmutex here. */
    pmutex = (pthread_mutex_t *) mmap(NULL, sizeof(pthread_mutex_t), PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS,
                                      -1, 0);

/* Initialise mutex. */
    pthread_mutex_init(pmutex, &attrmutex);

    const int num_children = 10;

    pid_t id;
    for (auto i = 0; i < num_children; ++i) {
        id = fork();
        if (id < 0) {
            std::cerr << "Could not fork the main process" << std::endl;
            std::exit(-1);
        }
        if (id == 0) {
            pthread_mutex_lock(pmutex);
            for (auto k = 0; k < 100; ++k) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                std::cout << i << k << std::endl;
            }
            pthread_mutex_unlock(pmutex);
            std::exit(1);
        }
    }

    if (id > 0) {
        //main_process



        int status;
        for (auto i = 0; i < num_children; ++i) {
            wait(&status);
        }
    }




    /* Clean up. */
    pthread_mutex_destroy(pmutex);
    pthread_mutexattr_destroy(&attrmutex);
}

int main(int argl, const char **argc) {

    initialize();
/*
    use_classical(true);
    std::vector<Position> positions;
    Board board;
    TT.resize(21);
    board = Position::getStartPosition();
    Utilities::createNMoveBook(std::back_inserter(positions), 3, board, -1000, 1000);
    Utilities::write_to_binary<Position>(positions.begin(),positions.end(),"/home/robin/DarkHorse/Training/Positions/train3.pos",std::ios::app);
    std::cout<<"Positions: "<<positions.size()<<std::endl;
*/


   remove_duplicates<Sample>("/home/robin/DarkHorse/Training/TrainData/test100",
                              "/home/robin/DarkHorse/Training/TrainData/test100removed");


/*    for (auto i = 0; i < 3; ++i) {
        for (auto j = 0; j < 2; ++j) {
            uint32_t reg = big_region1 << (i * 8 + j);
            temp.BP = reg;
            temp.printPosition();
        }
    }*/





       std::vector<Sample> opening;
       std::vector<Sample> ending;

       std::vector<Sample> data;

       Utilities::read_binary<Sample>(std::back_inserter(data),
                                      "/home/robin/DarkHorse/Training/TrainData/test100removed");

       for (Sample s : data) {
           auto num = Bits::pop_count(s.position.WP | s.position.BP);
           if (num <= 8) {
               ending.emplace_back(s);
           } else {
               opening.emplace_back(s);
           }
       }
       Utilities::write_to_binary<Sample>(opening.begin(), opening.end(),
                                          "/home/robin/DarkHorse/Training/TrainData/test100open.train");
       Utilities::write_to_binary<Sample>(ending.begin(), ending.end(),
                                          "/home/robin/DarkHorse/Training/TrainData/test100end.train");
















    Generator generator("master3", "train3.pos", "/home/robin/DarkHorse/Training/TrainData/test100");
    generator.set_num_games(10000000);
    generator.set_hash_size(20);
    generator.set_parallelism(96);
    generator.set_time(100);
    generator.startx();














//testing2 should be using the next depth3testupdate





/*
    network.load("testb");
    network.addLayer(Layer{120, 256});
    network.addLayer(Layer{256, 32});
    network.addLayer(Layer{32, 32});
    network.addLayer(Layer{32, 1});

    network.init();
    network2.load("testc");
    network2.addLayer(Layer{120, 256});
    network2.addLayer(Layer{256, 32});
    network2.addLayer(Layer{32, 32});
    network2.addLayer(Layer{32, 1});

    network2.init();

    std::vector<Sample> test_positions;
    Utilities::read_binary<Sample>(std::back_inserter(test_positions),
                                   "/home/robin/DarkHorse/Training/TrainData/small_dataset");

    int counter =0;
    for(Sample s : test_positions){
        if(counter>=100)
            break;
        s.position.printPosition();
        std::cout<<"Eval"<<network.compute_incre_forward_pass(s.position)<<std::endl;
        counter++;
    }
*/

/*
    Match engine_match("testcrazy", "master3");
    engine_match.setTime(100);
    engine_match.setMaxGames(100000);
    engine_match.setNumThreads(5);
    engine_match.setHashSize(21);
    engine_match.start()*/;


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



    /* std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
     Trainer trainer("/home/robin/DarkHorse/Training/TrainData/test100removed");
     trainer.setLearningRate(15000);
     trainer.setEpochs(1000);
     trainer.setl2Reg(0.000000000000);
     trainer.setCValue(-2e-3);
     trainer.startTune();
     auto loss = trainer.calculateLoss();
     std::cout << "Loss: " << loss << std::endl;
 */




    return 0;
}
