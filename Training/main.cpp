#include <iostream>
#include "Match.h"
#include "Generator.h"
#include <ostream>
#include <iterator>
#include "Network.h"
#include <GameLogic.h>
#include <sys/mman.h>
#include <BloomFilter.h>
#include <Util/LRUCache.h>
#include <Util/Compress.h>
#include <regex>
#include <algorithm>
#include "Util/Book.h"
#include "CmdParser.h"
#include <BatchProvider.h>

int main(int argl, const char** argc) {

    CmdParser parser(argl, argc);
    parser.parse_command_line();

    if (parser.has_option("match"))
    {
        if (parser.has_option("engines") && parser.has_option("time"))
        {
            auto engines = parser.as<std::vector<std::string>>("engines");
            auto time = parser.as<std::vector<int>>("time");

            Match engine_match(engines[0], engines[1]);
            engine_match.set_time(time[0],time[1]);

            if (parser.has_option("num_games"))
            {
                auto num_games = parser.as<int>("num_games");
                engine_match.setMaxGames(num_games);
            }

            if(parser.has_option("networks")) {
                auto networks = parser.as<std::vector<std::string>>("networks");
                if (!networks[0].empty() && !networks[1].empty())
                {
                    engine_match.set_arg1("--network " + networks[0]);
                    engine_match.set_arg2("--network " + networks[1]);
                }
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
        auto net_file = parser.as<std::string>("network");
        auto time = parser.as<int>("time");

        Generator generator;
        generator.set_network(net_file);
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
        } else {
            generator.set_hash_size(20);
        }

        if (parser.has_option("buffer_clear_count"))
        {
            auto clear_count= parser.as<int>("buffer_clear_count");
            generator.set_buffer_clear_count(clear_count);
        } else {
            generator.set_buffer_clear_count(20);
        }

        if (parser.has_option("threads"))
        {
            auto num_threads = parser.as<int>("threads");
            generator.set_parallelism(num_threads);
        } else {
            generator.set_parallelism(std::max(1u,std::thread::hardware_concurrency()-1));
        }

        if (parser.has_option("piece_limit"))
        {
            auto piece_limit = parser.as<int>("piece_limit");
            generator.set_piece_limit(piece_limit);
        }
        else
        {
            generator.set_piece_limit(6);
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
}
