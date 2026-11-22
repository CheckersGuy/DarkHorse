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

      // Simulate CheckerBoard's time bank for fixed time per move (engine
      // matches): time saved on one move is returned on later moves.
      if (parser.has_option("time_bank")) {
        engine_match.set_time_bank(true);
      }

      // Opening book, e.g. --book ../Training/Positions/11manballots.pos
      if (parser.has_option("book")) {
        engine_match.setOpeningBook(parser.as<std::string>("book"));
      }

      // Random start offsets in the book (e.g. per SPSA iteration)
      if (parser.has_option("book_seed")) {
        engine_match.set_book_seed(
            std::stoull(parser.as<std::string>("book_seed")));
      }

      // Sequential test on game pairs: --sprt elo0 elo1 [alpha beta]
      // NOTE: the time controls we use (e.g. 20 ms per move) were chosen on the
      // local machine. On another machine (cloud) run Training/bench_parallel.sh
      // on both machines first and scale --time by the NPS ratio, otherwise the
      // engines search a different number of nodes per move.
      // (logistic Elo, alpha = beta = 0.05 by default). The match stops as
      // soon as H0 (elo0) or H1 (elo1) is accepted; --num_games is a cap.
      if (parser.has_option("sprt")) {
        auto v = parser.as<std::vector<std::string>>("sprt");
        if (v.size() != 2 && v.size() != 4) {
          std::cerr << "--sprt needs elo0 elo1 [alpha beta], e.g. --sprt 0 5"
                    << std::endl;
          return 1;
        }
        const double e0 = std::stod(v[0]), e1 = std::stod(v[1]);
        const double a = v.size() == 4 ? std::stod(v[2]) : 0.05;
        const double b = v.size() == 4 ? std::stod(v[3]) : 0.05;
        if (!(e1 > e0) || a <= 0.0 || a >= 0.5 || b <= 0.0 || b >= 0.5) {
          std::cerr << "--sprt needs elo1 > elo0 and 0 < alpha, beta < 0.5"
                    << std::endl;
          return 1;
        }
        engine_match.set_sprt(e0, e1, a, b);
        engine_match.setMaxGames(200000);
      }

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
      // Extra engine options without leading dashes (the match tool's own
      // parser splits on "--"): --opts1 hist_mode=1 hist_div=8192 becomes
      // "--hist_mode 1 --hist_div 8192" on engine 1's command line.
      for (int e = 1; e <= 2; ++e) {
        const std::string key = "opts" + std::to_string(e);
        if (!parser.has_option(key))
          continue;
        std::string args = e == 1 ? engine_match.get_arg1()
                                  : engine_match.get_arg2();
        for (const auto &kv : parser.as<std::vector<std::string>>(key)) {
          const auto eq = kv.find('=');
          if (eq == std::string::npos || eq == 0) {
            std::cerr << "--" << key << " expects key=value, got " << kv
                      << std::endl;
            return 1;
          }
          if (!args.empty())
            args += " ";
          args += "--" + kv.substr(0, eq) + " " + kv.substr(eq + 1);
        }
        (e == 1 ? engine_match.set_arg1(args) : engine_match.set_arg2(args));
        std::cout << "Engine " << e << " options: " << args << std::endl;
      }
      engine_match.start();
    }
  }

  return 0;
}
