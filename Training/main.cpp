#include "CmdParser.h"
#include "Match.h"
#include "MovePicker.h"
#include "Network.h"
#include "Position.h"
#include "Utilities.h"
#include "types.h"
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

  // loading the openings I want to use for data generation
  /* TT.resize(2);
   Statistics::mPicker.init();
   std::ofstream output("training.pos");
   std::ifstream open("../Training/Positions/drawbook.book");
   network.load_bucket("nopsqt.quant");

   std::string opening;
   while (std::getline(open, opening)) {
     TT.clear();
     Statistics::mPicker.clear_scores();
     std::cout << opening << std::endl;
     Board board(Position::pos_from_fen(opening));
     board.get_position().print_position();
     std::cout << "Test" << std::endl << std::endl;
     ;
     Utilities::createNMoveBook(output, 9, board, -120, 120);
   }

   return 0;
 */
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
