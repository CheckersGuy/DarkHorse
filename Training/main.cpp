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
#include <algorithm>
#include "Util/Book.h"
#include "BatchProvider.h"
int main() {


  //counting unique positions in my book


  //Book::create_train_file("/home/leagu/DarkHorse/Training/Positions/11manballots.pos","/home/leagu/DarkHorse/Training/Positions/train3.pos",2);
 
 /*  std::ifstream stream("/home/leagu/DarkHorse/Training/Positions/train2.pos");
  std::unordered_set<Position>hash;
  std::istream_iterator<Position>begin(stream);
  std::istream_iterator<Position>end;
  size_t counter=0;
  size_t piece_count =0;
  size_t total_count =0;
  std::for_each(begin,end,[&](Position pos){
    if(hash.find(pos)==hash.end()){
        hash.insert(pos);
        counter++;
        piece_count+=Bits::pop_count(pos.BP|pos.WP);
    }
    total_count++;
  });

  std::cout<<"Number of positions "<<total_count<<std::endl;
  std::cout<<"Number of unique positions "<<counter<<std::endl;
  std::cout<<"Average piece_count"<< ((double)piece_count)/((double)total_count)<<std::endl;
  
  return 0;
 */

 
      merge_temporary_files("/home/leagu/DarkHorse/Training/TrainData/", "/home/leagu/DarkHorse/Training/TrainData/");
     /*      auto count = count_unique_positions("/home/leagu/DarkHorse/Training/TrainData/weird4.train");
   std::cout<<"Unique Positions so far: "<<count.first<<std::endl;
   std::cout<<"Total Positions so far: "<<count.second<<std::endl;    */ 
     
/*    
    Generator generator("train3.pos", "weird8.train");
    generator.set_hash_size(21);
    generator.set_buffer_clear_count(1000);
    generator.set_parallelism(12);
    generator.set_time(10);
    generator.set_piece_limit(6);
    generator.set_max_position(1500000000ull);
    generator.start();  
          */
/* 

    BatchProvider provider("/home/leagu/DarkHorse/Training/TrainData/weird7formatted.train",1000000,1000);

    std::unique_ptr<float[]>results = std::make_unique<float[]>(1000);
    std::unique_ptr<float[]>inputs = std::make_unique<float[]>(1000*120);
    std::unique_ptr<int64_t[]>moves = std::make_unique<int64_t[]>(1000);

    provider.next(results.get(),moves.get(),inputs.get());

    return 0;
 */
                           
     Match engine_match("form17", "test10sgd");
     engine_match.setTime(100);
     engine_match.setMaxGames(30000);
     engine_match.setNumThreads(12);
     engine_match.setHashSize(20);
     engine_match.start();                
                                                 
     
       Trainer trainer("/home/leagu/DarkHorse/Training/TrainData/weird8formatted.train");
      trainer.set_learning_rate(2000);
      trainer.set_train_file_locat("trainer.state");
      
      trainer.set_weight_decay(0);
      trainer.set_decay(0.08);
      trainer.set_weights_path("test11sgd.weights");
      trainer.set_savepoint_step(10000000);
      trainer.set_epochs(300);
      trainer.set_c_value(2.0e-2);
      //trainer.load_trainer_state("trainer.state");
      trainer.start_tune();    

/*     std::ifstream stream("/home/leagu/DarkHorse/Training/TrainData/medium.train");
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
