#include <iostream>
#include "Match.h"
#include "Generator.h"
#include <ostream>
#include <iterator>
#include "Network.h"
#include <GameLogic.h>
#include <sys/mman.h>
#include <BloomFilter.h>
#include "Trainer.h"
#include <Util/LRUCache.h>
#include <Util/Compress.h>
#include <regex>

int main() {


    //initialize();

    //use_classical(true);

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








   // merge_temporary_files("/home/robin/DarkHorse/Training/TrainData/", "/home/robin/DarkHorse/Training/TrainData/");



/*
    auto result = count_unique_positions("/home/robin/DarkHorse/Training/TrainData/large.train");
    std::cout << "Unique: " << result.first << " Total: " << result.second << std::endl;

*/


/*
    Generator generator("train4.pos", "largelarge.train");
    generator.set_hash_size(20);
    generator.set_buffer_clear_count(10000);
    generator.set_parallelism(95);
    generator.set_time(10);
    generator.set_piece_limit(6);
    generator.set_max_position(5000000000ull);
    generator.start();
*/


     Match engine_match("form_network6", "form_network5");
     engine_match.setTime(100);
     engine_match.setMaxGames(100000);
     engine_match.setNumThreads(6);
     engine_match.setHashSize(20);
     engine_match.start();

/*

      Trainer trainer("/home/robin/DarkHorse/Training/TrainData/largelargexxxx.train");
      trainer.set_learning_rate(2);
      trainer.set_weight_decay(0.0);
      trainer.set_decay(0.2);
      trainer.set_weights_path("largelarge.weights");
      trainer.set_savepoint_step(10000000);
      trainer.set_epochs(30);
      trainer.set_c_value(-1e-3);
      trainer.start_tune();
*/





    /*std::ofstream stream("test3.data", std::ios::app);*/
/*
    std::ifstream stream("test4.data");


    convert_to_new("/home/robin/DarkHorse/Training/TrainData/small_dataset4.train","/home/robin/DarkHorse/Training/TrainData/new_format.train");

*/

//testing regular expressions



    return 0;
}
