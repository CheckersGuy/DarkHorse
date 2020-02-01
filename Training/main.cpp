#include <iostream>
#include <iomanip>
#include "boost/program_options.hpp"
#include "Match.h"
#include "Trainer.h"
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

namespace opt =boost::program_options;

int main(int argc, char *argv[]) {
    opt::options_description all("All options");
    opt::options_description match("Match");
    opt::options_description utility("Utility");
    opt::options_description training("Training");


    all.add_options()
            ("time", opt::value<std::vector<int>>()->multitoken(), "Time to be used in the match")
            ("book", opt::value<std::string>()->default_value("3move.pos"), "Opening file to be used")
            ("maxGames", opt::value<int>()->default_value(10000000), "number of games")
            ("threads", opt::value<int>()->default_value(1), "number of threads")
            ("hashSize", opt::value<int>()->default_value(18), "hashSize to be used")
            ("output", opt::value<std::string>(), "output file");

    match.add_options()
            ("match", opt::value<std::vector<std::string>>()->required()->multitoken(), "Engines");


    utility.add_options()("listEngines", "List of all available engines")
            ("listTrainData", "List all available trainData")
            ("listWeights", "List all available weights")
            ("remDup", opt::value<std::string>(), "remove duplicates of a gameFile");

    training.add_options()("losses", opt::value<std::string>(), "Weights to be used for losses calculation")
            ("data", opt::value<std::string>(), "data to be used for losses calculation");

    opt::options_description generate("Generation");

    generate.add_options()
            ("generate", opt::value<std::string>()->required(), "Engine");

    opt::options_description desc("help");

    desc.add_options()("help", "display all the options");


    all.add(match).add(desc).add(generate).add(utility).add(training);


    opt::variables_map vm;

    opt::store(opt::parse_command_line(argc, argv, all), vm);


    if (vm.count("help")) {
        std::cout << all << std::endl;
    }


    if (vm.count("match")) {

        std::cout << "Getting stuff to work out" << std::endl;

        const std::vector<std::string> &engines = vm["match"].as<std::vector<std::string>>();
        if (engines.size() > 2) {
            std::cerr << "To many engines for a match" << std::endl;
            exit(EXIT_FAILURE);
        }


        for (const std::string &engine : engines) {
            std::cout << engine << std::endl;
        }


        const std::string path_one("/home/robin/DarkHorse/Training/Engines/" + engines[0]);
        const std::string path_two("/home/robin/DarkHorse/Training/Engines/" + engines[1]);
        Match m(path_one, path_two);


        if (vm.count("book")) {
            m.setOpeningBook("Positions/" + vm["book"].as<std::string>());
        }
        if (vm.count("time")) {
            const std::vector<int> &timeVector = vm["time"].as<std::vector<int>>();
            if (timeVector.size() == 1) {
                m.setTime(timeVector[0]);
            }
        }
        if (vm.count("threads")) {
            m.setNumThreads(vm["threads"].as<int>());
        }
        if (vm.count("maxGames")) {
            m.setMaxGames(vm["maxGames"].as<int>());
        }

        std::cout << "Book: " << m.getOpeningBook() << std::endl;
        std::cout << "Threads: " << m.getNumThreads() << std::endl;
        std::cout << "MaxGames: " << m.getMaxGames() << std::endl;
        std::cout << "HashSize: " << vm["hashSize"].as<int>() << std::endl;

        m.start();

    }
    using namespace Training;










/*
    initialize();
    std::cout << "Final Run" << std::endl;

    std::vector<TrainingPos> data;

    loadGames(data, "/home/robin/Schreibtisch/TrainData/test3.game");


    std::cout << "Length: " << data.size() << std::endl;


    auto removeCl = [](TrainingPos pos) {
        if (__builtin_popcount(pos.pos.BP | pos.pos.WP) <= 4)
            return true;
        Board board;
        board = pos.pos;
        Line local;
        Value qStatic = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0);
        return isWin(qStatic);
    };

    data.erase(std::remove_if(data.begin(), data.end(), removeCl), data.end());




*/





/*
    std::cout << "Positions after erase: " << data.size() << std::endl;

    std::cout << "Starting " << std::endl;
    std::cout << "Decesive: " << std::count_if(data.begin(), data.end(), [](const TrainingPos &pos) {
        return (pos.result >0 || pos.result<0);
    });
    std::cout << std::endl;

    std::cout << "Draws: " << std::count_if(data.begin(), data.end(), [](const TrainingPos &pos) {
        return (pos.result ==0);
    });
    std::cout << std::endl;

*/






    Match engine_match("reading", "reading2");
    engine_match.setTime(300);
    engine_match.setHashSize(25);

    engine_match.start();




/*
    Trainer trainer(data);
    trainer.setLearningRate(180000);
    trainer.setEpochs(100);
    trainer.setl2Reg(0.000000000000);
    trainer.setCValue(-3e-5);
    trainer.startTune();

*/












    return 0;
}
