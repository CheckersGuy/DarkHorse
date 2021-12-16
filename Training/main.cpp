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
#include <PosStreamer.h>


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

#include <BatchProvider.h>

int main(int argl, const char **argc) {

    initialize();

    use_classical(true);


/*

    std::mt19937_64 generator(231231231ull);
    std::ifstream stream("/home/robin/DarkHorse/Training/TrainData/big_data_shuffl.train", std::ios::binary);
    std::ofstream ostream("/home/robin/DarkHorse/Training/TrainData/opening_shuffle.train", std::ios::binary);
    std::vector<Sample> samples;
    std::istream_iterator<Sample> begin(stream);
    std::istream_iterator<Sample> end;

    std::copy(begin, end, std::back_inserter(samples));
    std::shuffle(samples.begin(), samples.end(), generator);

    size_t counter = 0;
    std::copy_if(samples.begin(), samples.end(), std::ostream_iterator<Sample>(ostream),[&](Sample s) {
        if (Bits::pop_count(s.position.BP | s.position.WP) > 10)
            counter++;
        return Bits::pop_count(s.position.BP | s.position.WP) > 10;
    } );

    std::cout << "Num_Position: " << counter << std::endl;

*/
/*
    std::ifstream stream("/home/robin/DarkHorse/Training/Positions/11manballots.pos", std::ios::binary);
    std::ifstream stream2("/home/robin/DarkHorse/Training/Positions/train3.pos", std::ios::binary);
    std::ofstream ostream("/home/robin/DarkHorse/Training/Positions/train4.pos", std::ios::binary);
    std::istream_iterator<Position> begin1(stream);
    std::istream_iterator<Position> end1;

    std::istream_iterator<Position> begin2(stream2);
    std::istream_iterator<Position> end2;

    std::copy(begin1, end1, std::ostream_iterator<Position>(ostream));
    std::copy(begin2, end2, std::ostream_iterator<Position>(ostream));
*/



//number of samples: 200117




/*


    remove_duplicates("/home/robin/DarkHorse/Training/TrainData/bloomcloudrescored",
                      "/home/robin/DarkHorse/Training/TrainData/bloomcloudrescored");

    return 0;

*/
/*
std::ofstream out("/home/robin/DarkHorse/Training/TrainData/policyopen.samples",std::ios::binary);
std::ifstream stream("/home/robin/PycharmProjects/pythonProject2/TrainData/policy.samples",std::ios::binary);
std::istream_iterator<Sample>begin(stream);
std::istream_iterator<Sample>end;
std::copy_if(begin,end,std::ostream_iterator<Sample>(out),[](Sample s){
    return Bits::pop_count(s.position.BP | s.position.WP)>8;
});
return 0;

*/


/*


      Utilities::create_samples_from_games("/home/robin/DarkHorse/Training/TrainData/lastcheck.games", "/home/robin/DarkHorse/Training/TrainData/xxx.samples");
      return 0;
*/

    Generator generator("train4.pos", "/home/robin/DarkHorse/Training/TrainData/bigopenset.games");
    generator.set_hash_size(18);
    generator.set_buffer_clear_count(10000);
    generator.set_parallelism(96);
    generator.set_time(10);
    generator.set_max_position(1000000000ull);
    generator.startx();



/*

     Match engine_match("test4", "test3");
     engine_match.setTime(100);
     engine_match.setMaxGames(100000);
     engine_match.setNumThreads(14);
     engine_match.setHashSize(21);
     engine_match.start();

*/












/*
  BatchProvider provider("/home/robin/DarkHorse/Training/TrainData/lastcheck.samples",1000,10);

  for(auto i=0;i<100000;++i){
      std::array<float,120>inputs;
      std::array<float,1>results;
      provider.next(&results.front(),&inputs.front());
      float s = 0;
      for(auto k=0;k<120;++k)
          s+=inputs[k];
      if(s>10000){
          std::cout<<s<<std::endl;
          std::cout<<i<<std::endl;
          break;
      }
  }
*/

    // 0.190537  1e-4
    //0.188262   6e-4
    //0.188813 1e-3
    //0.127496

/*    std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
    Trainer trainer("/home/robin/DarkHorse/Training/TrainData/small_dataset_1m.train");
    trainer.setLearningRate(32000);
    trainer.setEpochs(30);
    trainer.setl2Reg(0.000000000000);
    trainer.setCValue(-1e-3);
    trainer.startTune();
    auto loss = trainer.calculateLoss();
    std::cout << "Loss: " << loss << std::endl;*/


    return 0;
}
