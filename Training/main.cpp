#include "CmdParser.h"
#include "Generator.h"
#include "Match.h"
#include "Network.h"
#include "Util/Book.h"
#include <BatchProvider.h>
#include <BloomFilter.h>
#include <GameLogic.h>
#include <Util/Compress.h>
#include <Util/LRUCache.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <ostream>
#include <regex>
#include <sys/mman.h>
#include "generator.pb.h"
#include <fstream>
#include "Position.h"
#include <string>

int main(int argl, const char **argc) {


  write_raw_data("/home/leagu/DarkHorse/Training/TrainData/testing.train");
  sort_raw_data("/home/leagu/DarkHorse/Training/TrainData/testing.train.raw" ,"/home/leagu/DarkHorse/Training/TrainData/shuffled.train.raw");
  return 0;

 // Book::create_train_file(
   //   "/home/leagu/DarkHorse/Training/Positions/11manballots.pos",
    //  "/home/leagu/DarkHorse/Training/Positions/train12.book", 6);        

  //GameStat stats;
  //get_game_stats("/home/leagu/DarkHorse/Training/TrainData/giga.train", stats);
  //std::cout<<stats<<std::endl;
  //return 0;
  //view_game("/home/leagu/DarkHorse/Training/TrainData/testing2.train", 2222);
  /*     std::ifstream
     stream("/home/leagu/DarkHorse/Training/TrainData/reinf.train");
      std::istream_iterator<Game>begin(stream);
      std::istream_iterator<Game>end;
      std::for_each(begin,end,[](Game g){
          std::vector<Position>positions;
          g.extract_positions(std::back_inserter(positions));
          for(auto pos : positions){
              pos.print_position();
          }
          std::cout<<"\n\n\n";

      });

      return 0;  */
  /*     for(auto i=0;i<50;++i){
      Encoding test;
      auto val = i%4;
      test.set_result(static_cast<Result>(val));
      std::cout<<val<<std::endl;
      std::cout<<(int)test.get_result()<<std::endl;

      }
      return 0; */
  //write_raw_data("/home/leagu/DarkHorse/Training/TrainData/testme.train");
   //sort_raw_data("/home/leagu/DarkHorse/Training/TrainData/testme.train.raw");
   //return 0;
  

  CmdParser parser(argl, argc);
  parser.parse_command_line();
  

  if(parser.has_option("create_raw")){
    auto input_file = parser.as<std::string>("create_raw");
    auto path = "/home/leagu/DarkHorse/Training/TrainData/"+input_file;
    create_shuffled_raw(path);
  }

  if (parser.has_option("match")) {
    if (parser.has_option("engines") && parser.has_option("time")) {
      auto engines = parser.as<std::vector<std::string>>("engines");
      auto time = parser.as<std::vector<int>>("time");

      Match engine_match(engines[0], engines[1]);
      engine_match.set_time(time[0], time[1]);

      if (parser.has_option("num_games")) {
        auto num_games = parser.as<int>("num_games");
        engine_match.setMaxGames(num_games);
      }

      if (parser.has_option("networks")) {
        auto networks = parser.as<std::vector<std::string>>("networks");
        if (!networks[0].empty() && !networks[1].empty()) {
          engine_match.set_arg1("--network " + networks[0]);
          engine_match.set_arg2("--network " + networks[1]);
        }
      }

      if (parser.has_option("threads")) {
        auto num_threads = parser.as<int>("threads");
        engine_match.setNumThreads(num_threads);
      } else {
        engine_match.setNumThreads(
            std::max(1u, std::thread::hardware_concurrency() - 1));
      }
      if (parser.has_option("hash_size")) {
        auto hash_size = parser.as<int>("hash_size");
        engine_match.setHashSize(hash_size);
      } else {
        engine_match.setHashSize(21);
      }
      engine_match.start();
    }
  }

  if (parser.has_option("generate") && parser.has_option("network") &&
      parser.has_option("time")) {
    merge_temporary_files("/home/leagu/DarkHorse/Training/TrainData/",
                          "/home/leagu/DarkHorse/Training/TrainData/");
    // network not yet supported
    auto net_file = parser.as<std::string>("network");
    auto time = parser.as<int>("time");

    Generator generator;
    generator.set_network(net_file);
    if (parser.has_option("book")) {
      auto book = parser.as<std::string>("book");
      generator.set_book(book);
    } else {
      generator.set_book("train8.pos");
    }

    if (parser.has_option("output")) {
      auto output = parser.as<std::string>("output");
      generator.set_output(output);
    } else {
      generator.set_output("reinf.train");
    }

    if (parser.has_option("hash_size")) {
      auto hash_size = parser.as<int>("hash_size");
      generator.set_hash_size(hash_size);
    } else {
      generator.set_hash_size(20);
    }

    if (parser.has_option("buffer_clear_count")) {
      auto clear_count = parser.as<int>("buffer_clear_count");
      generator.set_buffer_clear_count(clear_count);
    } else {
      generator.set_buffer_clear_count(20);
    }

    if (parser.has_option("threads")) {
      auto num_threads = parser.as<int>("threads");
      generator.set_parallelism(num_threads);
    } else {
      generator.set_parallelism(
          std::max(1u, std::thread::hardware_concurrency() - 1));
    }

    if (parser.has_option("piece_limit")) {
      auto piece_limit = parser.as<int>("piece_limit");
      generator.set_piece_limit(piece_limit);
    } else {
      generator.set_piece_limit(6);
    }

    generator.set_time(time);

    if (parser.has_option("max_games")) {
      auto max_games = parser.as<int>("max_games");
      generator.set_max_games(max_games);
    }
    generator.start();
    merge_temporary_files("/home/leagu/DarkHorse/Training/TrainData/",
                          "/home/leagu/DarkHorse/Training/TrainData/");
  }

  return 0;
}
