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
int main() {

/*  
 
     merge_temporary_files("/home/leagu/DarkHorse/Training/TrainData/", "/home/leagu/DarkHorse/Training/TrainData/");
    auto count = count_unique_positions("/home/leagu/DarkHorse/Training/TrainData/weird4.train");
   std::cout<<"Unique Positions so far: "<<count.first<<std::endl;
   std::cout<<"Total Positions so far: "<<count.second<<std::endl; 
   
  */
/*
    auto result = count_unique_positions("/home/robin/DarkHorse/Training/TrainData/large.train");
    std::cout << "Unique: " << result.first << " Total: " << result.second << std::endl;

/* */   
/*      Generator generator("train.pos", "weird4.train");
    generator.set_hash_size(21);
    generator.set_buffer_clear_count(1000);
    generator.set_parallelism(14);
    generator.set_time(30);
    generator.set_piece_limit(6);
    generator.set_max_position(150000000ull);
    generator.start();    */
              
    
    //Book::create_train_file("/home/leagu/DarkHorse/Training/Positions/11manballots.pos","/home/leagu/DarkHorse/Training/Positions/train.pos",3);
  
      Match engine_match("test7sgd", "newtryx");
     engine_match.setTime(30);
     engine_match.setMaxGames(30000);
     engine_match.setNumThreads(12);
     engine_match.setHashSize(20);
     engine_match.start();         
                       
  
       Trainer trainer("/home/leagu/DarkHorse/Training/TrainData/weird4formatted.train");
      trainer.set_learning_rate(6000);
      trainer.set_train_file_locat("trainer.state");
      
      trainer.set_weight_decay(0);
      trainer.set_decay(0.08);
      trainer.set_weights_path("test7bla.weights");
      trainer.set_savepoint_step(10000000);
      trainer.set_epochs(1000);
      trainer.set_c_value(2.24e-2);
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
