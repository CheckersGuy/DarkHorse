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


    //testing new policy format

/*     std::ifstream stream("/home/leagu/DarkHorse/Training/TrainData/medium.train");
    if(!stream.good())
        std::exit(-1);

    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game>end;
    std::vector<Game> games;
    std::copy(begin,end,std::back_inserter(games));
    size_t counter  =0;
    std::cout<<"Start"<<std::endl;
    auto t1 = std::chrono::high_resolution_clock::now();
    std::for_each(games.begin(),games.end(),[&](Game game){
        for(auto pos : game){
            counter++;
        }
    });
   auto t2 = std::chrono::high_resolution_clock::now();
   auto dur = (t2-t1).count();
   std::cout<<"Duration: "<<(dur/1000000)<<std::endl;
   std::cout<<"Counter: "<<counter<<std::endl; */

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






     Match engine_match("largelarge", "largelarge");
     engine_match.setTime(10);
     engine_match.setMaxGames(30000);
     engine_match.setNumThreads(6);
     engine_match.setHashSize(21);
     engine_match.start();  
    
/*  
       Trainer trainer("/home/leagu/DarkHorse/Training/TrainData/verylargexxxx.train");
      trainer.set_learning_rate(5);
      trainer.set_weight_decay(0.0);
      trainer.set_decay(0.1);
      trainer.set_weights_path("bla.weights");
      trainer.set_savepoint_step(10000000);
      trainer.set_epochs(1);
      trainer.set_c_value(-2e-3);
      trainer.start_tune();   
 */

        //epoch: 3
        // 0.157794

/*     std::ifstream stream("/home/leagu/DarkHorse/Training/TrainData/verylargexxxx.train");
    Game g;
    while(stream>>g){
        std::vector<Sample> vec;
        std::vector<Sample> vec2;
        g.extract_samples_test(std::back_inserter(vec));
        g.extract_samples(std::back_inserter(vec2));

        for(int i=0;i<vec2.size();++i){
            Sample s = vec[i];
            Sample s2 = vec2[i];
            if(s!=s2){
                std::cout<<std::endl;
                std::cout<<"First"<<std::endl;
                std::cout<<"Result: "<<s.result<<std::endl;
                s.position.print_position();
                  std::cout<<"Second"<<std::endl;
                std::cout<<"Result: "<<s2.result<<std::endl;
                s2.position.print_position();
                

            }

        }

    }
  */

    return 0;
}
