#include <iostream>
#include <iomanip>
#include "boost/program_options.hpp"
#include "Match.h"
#include "Generator.h"
#include "Trainer.h"
#include "boost/filesystem.hpp"
#include "boost/foreach.hpp"

namespace fs =boost::filesystem;
namespace opt =boost::program_options;


std::string formatByteSize(const uint64_t size) {
    //Formats the size given in byte to MB,GB...
    std::string output;
    const double sizes[] = {1e12, 1e9, 1e6, 1e3};
    std::vector<std::string> names = {"TB", "GB", "MB", "KB"};
    auto it = names.begin();
    for (auto e : sizes) {
        if (size >= e) {

            output += std::to_string(((double) (size)) / e);
            output += "" + *it;
            break;
        }
        it++;
    }
    return output;
}

void listDirectory(fs::path &myPath) {
    fs::directory_iterator it(myPath), eod;
    BOOST_FOREACH(fs::path const &p, std::make_pair(it, eod)) {
                    constexpr int indent = 25;
                    if (fs::is_regular_file(p)) {
                        std::cout << p.filename().string() << " ";
                        for (int i = 0; i < indent - p.filename().string().size(); ++i)
                            std::cout << " ";

                        std::cout << "Size:";
                        std::cout << formatByteSize(fs::file_size(p));
                        std::cout << std::endl;
                    }
                }
}

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

    if (vm.count("listEngines")) {
        fs::path myPath("/home/robin/Checkers/Training/Engines");
        listDirectory(myPath);
    }

    if (vm.count("listWeights")) {
        fs::path myPath("/home/robin/Checkers/Training/Weights");
        listDirectory(myPath);
    }

    if (vm.count("listTrainData")) {
        fs::path myPath("/home/robin/Checkers/Training/TrainData");
        listDirectory(myPath);
    }

    if (vm.count("match")) {
        const std::vector<std::string> &engines = vm["match"].as<std::vector<std::string>>();
        if (engines.size() > 2) {
            std::cerr << "To many engines for a match" << std::endl;
            exit(EXIT_FAILURE);
        }


        for (const std::string &engine : engines) {
            std::cout << engine << std::endl;
        }


        Engine one("/home/robin/DarkHorse/Training/cmake-build-debug/Engines/" + engines[0]);
        Engine two("/home/robin/DarkHorse/Training/cmake-build-debug/Engines/"  + engines[1]);
        Match match(one, two);
        one.setHashSize(vm["hashSize"].as<int>());
        two.setHashSize(vm["hashSize"].as<int>());


        if (vm.count("book")) {
            match.setOpeningBook("Positions/" + vm["book"].as<std::string>());
        }
        if (vm.count("time")) {
            const std::vector<int> &timeVector = vm["time"].as<std::vector<int>>();
            if (timeVector.size() == 1) {
                one.setTimePerMove(timeVector[0]);
                two.setTimePerMove(timeVector[0]);
            } else if (timeVector.size() == 2) {
                one.setTimePerMove(timeVector[0]);
                two.setTimePerMove(timeVector[1]);
            } else {
                one.setTimePerMove(50);
                two.setTimePerMove(50);
            }
        }
        if (vm.count("threads")) {
            match.setNumThreads(vm["threads"].as<int>());
        }
        if (vm.count("maxGames")) {
            match.setMaxGames(vm["maxGames"].as<int>());
        }

        std::cout << "Book: " << match.getOpeningBook() << std::endl;
        std::cout << "Time: " << one.getTimePerMove() << " | " << two.getTimePerMove() << std::endl;
        std::cout << "Threads: " << match.getNumThreads() << std::endl;
        std::cout << "MaxGames: " << match.getMaxGames() << std::endl;
        std::cout << "HashSize: " << vm["hashSize"].as<int>() << std::endl;

        match.start();

    }

    if (vm.count("generate")) {


        Zobrist::initializeZobrisKeys();
        std::string engine = vm["generate"].as<std::string>();
        std::string path = "Engines/" + engine;
        auto &timeVector = vm["time"].as<std::vector<int>>();
        int time = timeVector[0];

        Engine myEngine(path);
        myEngine.setHashSize(vm["hashSize"].as<int>());
        Generator gen(myEngine, "Positions/genBook6.pos", "TrainData/" + vm["output"].as<std::string>());
        gen.setThreads(vm["threads"].as<int>());
        gen.setMaxGames(vm["maxGames"].as<int>());
        gen.setTime(100);


        std::cout << "Time: " << myEngine.getTimePerMove() << std::endl;
        std::cout << "Threads: " << gen.getThreads() << std::endl;
        std::cout << "MaxGames: " << gen.getMaxGames()<< std::endl;
        std::cout << "HashSize: " << vm["hashSize"].as<int>() << std::endl;
        std::cout<<std::endl;
        std::cout<<std::endl;

        gen.start();
    }
    using namespace Training;

    initialize();
    std::cout<<"Final Run"<<std::endl;

    std::vector<TrainingPos> data;

    loadGames(data, "/home/robin/DarkHorse/Training/TrainData/test3.game");


    std::cout << "Length: " << data.size() << std::endl;


/*
    auto removeCl = [](TrainingPos pos) {

        if(__builtin_popcount(pos.pos.BP|pos.pos.WP)<=4)
            return true;

        Board board;
        board=pos.pos;
        Line local;
        Value qStatic = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0);
        return qStatic.isWinning();
    };

    data.erase(std::remove_if(data.begin(),data.end(),removeCl),data.end());
*/


    std::cout << "Positions after erase: " << data.size() << std::endl;

    std::cout << "Starting " << std::endl;


    Trainer trainer(data);
    // 0.0399091
    trainer.setLearningRate(180000);
    trainer.setEpochs(100);
    trainer.setl2Reg(0.000000000000);
    trainer.setCValue(-6e-5);
    trainer.startTune();





    return 0;
}
