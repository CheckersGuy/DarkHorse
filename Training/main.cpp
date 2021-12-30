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
#include "Util/File.h"

int main(int argl, const char **argc) {

    initialize();

    use_classical(true);

    /*   std::ifstream stream("/home/robin/DarkHorse/Training/TrainData/endgame_shuffle.train", std::ios::binary);
       std::ofstream stream_out("/home/robin/DarkHorse/Training/TrainData/endgame.train",std::ios::binary);
       std::istream_iterator<Sample> begin(stream);
       std::istream_iterator<Sample> end{};
   */
    /*   std::for_each(begin, end, [](Sample s) {
           int result = static_cast<int>(s.result);
           if(result == -1){
              s.result = BLACK_WON;
           }else if(result ==1){
               s.result = WHITE_WON;
           }else if(result ==0){
               s.result = DRAW;
           }else{
               s.result = UNKNOWN;
           }
           std::cout << result << std::endl;
       });
       */
    /*  std::transform(begin,end,std::ostream_iterator<Sample>(stream_out),[](Sample s){
          int result = static_cast<int>(s.result);
          Sample copy = s;
          if(result == -1){
              copy.result = BLACK_WON;
          }else if(result ==1){
              copy.result = WHITE_WON;
          }else if(result ==0){
              copy.result = DRAW;
          }else{
              copy.result = UNKNOWN;
          }
          return copy;
          });

  */


    /*   merge_files<Sample>({"/home/robin/DarkHorse/Training/TrainData/bigopenset.train",
                            "/home/robin/DarkHorse/Training/TrainData/endgame.train"},
                           "/home/robin/DarkHorse/Training/TrainData/patt.train");

   */


    /*   std::ifstream stream("/home/robin/DarkHorse/Training/TrainData/patt.train", std::ios::binary);
       std::ofstream ostream("/home/robin/DarkHorse/Training/TrainData/patt_val.train", std::ios::binary);
       std::istream_iterator<Sample> begin(stream);
       std::istream_iterator<Sample> end;
       std::copy_n(begin,100000000, std::ostream_iterator<Sample>(ostream));

       return 0;*/





/*
    Generator generator("train4.pos", "/home/robin/DarkHorse/Training/TrainData/small_dataset2.train");
    generator.set_hash_size(18);
    generator.set_buffer_clear_count(10000);
    generator.set_parallelism(16);
    generator.set_time(10);
    generator.set_piece_limit(10);
    generator.set_max_position(1000000000ull);
    generator.startx();

*/









/*



     Match engine_match("smalltest", "master");
     engine_match.setTime(100);
     engine_match.setMaxGames(100000);
     engine_match.setNumThreads(12);
     engine_match.setHashSize(21);
     engine_match.start();

*/









    std::cout << "NonZeroWeights: " << gameWeights.numNonZeroValues() << std::endl;
    Trainer trainer("/home/robin/DarkHorse/Training/TrainData/small_dataset2.train");
    trainer.setLearningRate(50000);
    trainer.set_decay(0.05);
    trainer.set_weights_path("smalltest.weights");
    trainer.set_savepoint_step(10000000);
    trainer.setEpochs(2000);
    trainer.setCValue(-5e-3);
    trainer.startTune();




    return 0;
}
