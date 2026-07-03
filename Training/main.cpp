#include "CmdParser.h"
#include "Match.h"
#include "Network.h"
#include "Position.h"
#include <GameLogic.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <ostream>
#include <regex>
#include <string>
#include <sys/mman.h>

// reading the training data here, will be moved to a utility file later
// computing a histogram of the activations
// computing average number of nnz-blocks

// Try algorithms to decrease nnz

// 1. Simply sorting the activations by the histogram
// 2. measuring the results and implement support for arbitrary permutations
// 3. Look at how stockfish does this

int main(int argl, const char **argc) {

  CmdParser parser;
  parser.parse(argl, argc);

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
