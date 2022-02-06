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
    Generator generator("train4.pos", "/home/robin/DarkHorse/Training/TrainData/small_dataset6.games");
    generator.set_hash_size(18);
    generator.set_buffer_clear_count(10000);
    generator.set_parallelism(14);
    generator.set_time(20);
    generator.set_piece_limit(10);
    generator.set_max_position(5000000ull);
    generator.startx();*/



    //working on a new storage format for opening positions
    //Positions either start from the regular start position or from
    //one of the 2500 11 man ballots !!
    //which means the first bits are used as an index




     Match engine_match("test", "test");
     engine_match.setTime(100);
     engine_match.setMaxGames(100000);
     engine_match.setNumThreads(12);
     engine_match.setHashSize(21);
     engine_match.start();






/*    Trainer trainer("/home/robin/DarkHorse/Training/TrainData/small_dataset4.val");
    trainer.set_learning_rate(50000);
    trainer.set_weight_decay(0.0);
    trainer.set_decay(0.07);
    trainer.set_weights_path("bla.weights");
    trainer.set_savepoint_step(10000000);
    trainer.set_epochs(1000);
    trainer.set_c_value(-1e-3);
    trainer.start_tune();*/

    //creating a subset of my dataset for validation data





    return 0;
}
