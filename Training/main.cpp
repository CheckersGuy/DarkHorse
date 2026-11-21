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

      // Search threads per engine (Lazy SMP), e.g. --search_threads 1 2.
      // Not to be confused with --threads, the number of games run in
      // parallel. Passed to the engines as their --threads argument.
      if (parser.has_option("search_threads")) {
        auto search_threads = parser.as<std::vector<int>>("search_threads");
        if (search_threads.size() != 2 || search_threads[0] < 1 ||
            search_threads[1] < 1) {
          std::cerr << "--search_threads needs two values >= 1, e.g. "
                       "--search_threads 1 2"
                    << std::endl;
          return 1;
        }
        engine_match.set_arg1("--threads " + std::to_string(search_threads[0]));
        engine_match.set_arg2("--threads " + std::to_string(search_threads[1]));

        const unsigned cores = std::thread::hardware_concurrency();
        const unsigned busy =
            engine_match.getNumThreads() *
            std::max(search_threads[0], search_threads[1]);
        std::cout << "Search threads: engine 1 = " << search_threads[0]
                  << ", engine 2 = " << search_threads[1] << std::endl;
        if (cores != 0 && busy > cores) {
          std::cout << "Warning: up to " << busy << " busy threads ("
                    << engine_match.getNumThreads()
                    << " parallel games) on " << cores
                    << " hardware threads; reduce --threads to avoid "
                       "distorting the result"
                    << std::endl;
        }
      }
      // Policy logit scale per engine, e.g. --policy_scale 2.2 1.0.
      // Passed to the engines as their --policy_scale argument.
      if (parser.has_option("policy_scale")) {
        auto scales = parser.as<std::vector<std::string>>("policy_scale");
        if (scales.size() != 2) {
          std::cerr << "--policy_scale needs two values, e.g. "
                       "--policy_scale 2.2 1.0"
                    << std::endl;
          return 1;
        }
        auto join = [](const std::string &a, const std::string &b) {
          return a.empty() ? b : a + " " + b;
        };
        engine_match.set_arg1(
            join(engine_match.get_arg1(), "--policy_scale " + scales[0]));
        engine_match.set_arg2(
            join(engine_match.get_arg2(), "--policy_scale " + scales[1]));
        std::cout << "Policy scale: engine 1 = " << scales[0]
                  << ", engine 2 = " << scales[1] << std::endl;
      }
      engine_match.start();
    }
  }

  return 0;
}
