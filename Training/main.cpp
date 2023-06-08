#include "CmdParser.h"
#include "Match.h"
#include "Network.h"
#include "Position.h"
#include "Util/Book.h"
#include "generator.pb.h"
#include <BatchProvider.h>
#include <BloomFilter.h>
#include <GameLogic.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ostream>
#include <regex>
#include <string>
#include <sys/mman.h>

int main(int argl, const char **argc) {

  // write_raw_data("/home/leagu/DarkHorse/Training/TrainData/windowmaster.train");
  // sort_raw_data(
  //     "/home/leagu/DarkHorse/Training/TrainData/windowmaster.train.raw",
  //     "/home/leagu/DarkHorse/Training/TrainData/windowmasterremoved.train");
  /*
    Book::create_train_file(
        "/home/leagu/DarkHorse/Training/Positions/11manballots.pos",
        "/home/leagu/DarkHorse/Training/Positions/train12.book", 9);

  */
  // return 0;
  // GameStat stats;
  // get_game_stats("/home/leagu/DarkHorse/Training/TrainData/ultimate.train",
  //              stats);
  // std::cout << stats << std::endl;
  // return 0;
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
  // write_raw_data("/home/leagu/DarkHorse/Training/TrainData/testme.train");
  // sort_raw_data("/home/leagu/DarkHorse/Training/TrainData/testme.train.raw");
  // return 0;

  CmdParser parser(argl, argc);
  parser.parse_command_line();

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

  return 0;
}
