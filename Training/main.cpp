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
#include "CmdParser.h"
int main(int argl, const char** argc) {

  CmdParser parser(argl, argc);
  parser.parse_command_line();


/* 
  std::ifstream stream("/home/leagu/DarkHorse/Training/TrainData/reinf.train");

  std::istream_iterator<Game>begin(stream);
  std::istream_iterator<Game>end;

  std::for_each(begin,end,[](Game g){
    for(auto p : g){
      p.print_position();
    }
    std::cout<<std::endl;
    std::cout<<std::endl;
    
  });
  return 0;
   */


  if (parser.has_option("match"))
  {
    if (parser.has_option("engines") && parser.has_option("time"))
    {
      auto engines = parser.as<std::vector<std::string>>("engines");
      auto time = parser.as<std::vector<int>>("time");

      Match engine_match(engines[0], engines[1]);
      engine_match.setTime(time[0]);

      if (parser.has_option("num_games"))
      {
        auto num_games = parser.as<int>("num_games");
        engine_match.setMaxGames(num_games);
      }

      if (parser.has_option("threads"))
      {
        auto num_threads = parser.as<int>("threads");
        engine_match.setNumThreads(num_threads);
      }
      else
      {
        engine_match.setNumThreads(std::max(1u, std::thread::hardware_concurrency() - 1));
      }
      if (parser.has_option("hash_size"))
      {
        auto hash_size = parser.as<int>("hash_size");
        engine_match.setHashSize(hash_size);
      }
      else
      {
        engine_match.setHashSize(21);
      }
      engine_match.start();
    }
  }

  if (parser.has_option("generate") && parser.has_option("network") && parser.has_option("time"))
  {
    merge_temporary_files("/home/leagu/DarkHorse/Training/TrainData/", "/home/leagu/DarkHorse/Training/TrainData/");
    //network not yet supported
    auto network = parser.as<std::string>("network");
    auto time = parser.as<int>("time");

    Generator generator;

    if (parser.has_option("book"))
    {
      auto book = parser.as<std::string>("book");
      generator.set_book(book);
    }
    else
    {
      generator.set_book("train6.pos");
    }

    if (parser.has_option("output"))
    {
      auto output = parser.as<std::string>("output");
      generator.set_output(output);
    }
    else
    {
      generator.set_output("reinf.train");
    }

       if (parser.has_option("hash_size"))
      {
        auto hash_size = parser.as<int>("hash_size");
        generator.set_hash_size(hash_size);
      }else{
        generator.set_hash_size(20);
      }

      if (parser.has_option("buffer_clear_count"))
      {
        auto clear_count= parser.as<int>("buffer_clear_count");
         generator.set_buffer_clear_count(clear_count);
      }else{
         generator.set_buffer_clear_count(20);
      }

      if (parser.has_option("threads"))
      {
        auto num_threads = parser.as<int>("threads");
          generator.set_parallelism(num_threads);
      }else{
           generator.set_parallelism(std::max(1u,std::thread::hardware_concurrency()-1));
      }

      if (parser.has_option("piece_limit"))
      {
        auto piece_limit = parser.as<int>("piece_limit");
        generator.set_piece_limit(piece_limit);
      }
      else
      {
       generator.set_piece_limit(5);
      }

    generator.set_time(time);

     if (parser.has_option("max_games"))
      {
        auto max_games = parser.as<int>("max_games");
        generator.set_max_games(max_games);
      }
    generator.start();
    merge_temporary_files("/home/leagu/DarkHorse/Training/TrainData/", "/home/leagu/DarkHorse/Training/TrainData/");
  }

  return 0;



  /*  Book::create_train_file("/home/leagu/DarkHorse/Training/Positions/11manballots.pos","/home/leagu/DarkHorse/Training/Positions/train6.pos",5);
   return 0;
  */
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
 /*  */
  /*      merge_temporary_files("/home/leagu/DarkHorse/Training/TrainData/", "/home/leagu/DarkHorse/Training/TrainData/");
    auto count = count_unique_positions("/home/leagu/DarkHorse/Training/TrainData/reinf.train");
    std::cout << "Unique Positions so far: " << count.first << std::endl;
    std::cout << "Total Positions so far: " << count.second << std::endl;    */
  //return 0;  
     
  /*    Generator generator("train6.pos", "reinf.train");
  generator.set_hash_size(20);
  generator.set_buffer_clear_count(20);
  generator.set_parallelism(14);
  generator.set_time(10);
  generator.set_piece_limit(5);
  generator.set_max_position(1500000000ull);
  generator.start();        */        
             
    Match engine_match("bigagain2", "reinfnet");
  engine_match.setTime(100);
  engine_match.setMaxGames(300000);
  engine_match.setNumThreads(14);
  engine_match.setHashSize(21);
  engine_match.start();  
 
         Trainer trainer("/home/leagu/DarkHorse/Training/TrainData/weird9formatted.train");
       trainer.set_learning_rate(8000);
       trainer.set_train_file_locat("trainer.state");

       trainer.set_weight_decay(0);
       trainer.set_decay(0.08);
       trainer.set_weights_path("test12sgd.weights");
       trainer.set_savepoint_step(10000000);
       trainer.set_epochs(300);
       trainer.set_c_value(2.0e-2);
       //trainer.load_trainer_state("trainer.state");
       trainer.start_tune();   
  return 0;
}
