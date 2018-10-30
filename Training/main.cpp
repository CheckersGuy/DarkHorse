#include <iostream>
#include <iomanip>
#include "boost/program_options.hpp"
#include "Match.h"
#include "Generator.h"
#include "SimplePool.h"
#include "Trainer.h"
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/copy.hpp>
#include <boost/iostreams/filter/bzip2.hpp>


int main(int argc, char *argv[]) {


    namespace opt =boost::program_options;

    opt::options_description match("Match");

    match.add_options()
            ("match", opt::value<std::vector<std::string>>()->required()->multitoken(), "Engines")
            ("time", opt::value<std::vector<int>>(), "Time to be used in the match")
            ("book", opt::value<std::string>()->default_value("3move.pos"), "Opening file to be used")
            ("maxGames", opt::value<int>()->default_value(100), "number of games")
            ("threads", opt::value<int>()->default_value(1), "number of threads");


    opt::options_description generate("Generation");

    generate.add_options()
            ("generate", opt::value<std::string>()->required(), "Engine")
            ("time", opt::value<int>()->default_value(100), "Time to be used in the generation")
            ("book", opt::value<std::string>()->default_value("testPositions2.pos"), "Opening file to be used")
            ("output", opt::value<std::string>()->default_value("training.game"), "output file")
            ("maxGames", opt::value<int>()->default_value(100), "number of games")
            ("threads", opt::value<int>()->default_value(1), "number of threads");


    opt::options_description desc("help");

    desc.add_options()("help", "display all the options");

    opt::options_description all("All options");

    all.add(match).add(desc);


    opt::variables_map vm;

    opt::store(opt::parse_command_line(argc, argv, all), vm);


    if (vm.count("help")) {
        std::cout << match << std::endl;
        std::cout << generate << std::endl;
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


        Engine one("Engines/" + engines[0]);
        Engine two("Engines/" + engines[1]);
        Match match(one, two);


        if (vm.count("book")) {
            match.setOpeningBook("Positions/" + vm["book"].as<std::string>());
        }
        if (vm.count("time")) {
            std::vector<int> time = vm["time"].as<std::vector<int>>();
            if (time.size() == 1) {
                match.setTime(time[0]);
            } else if (time.size() == 2) {
                //to be done
            }
        }
        if (vm.count("threads")) {
            std::cout << "Test: " << vm["threads"].as<int>() << std::endl;
            match.setNumThreads(vm["threads"].as<int>());
        }
        if (vm.count("maxGames")) {
            std::cout << "Test: " << vm["maxGames"].as<int>() << std::endl;
            match.setMaxGames(vm["maxGames"].as<int>());
        }

        std::cout << "Book: " << match.getOpeningBook() << std::endl;
        std::cout << "Time: " << match.getTime() << std::endl;
        std::cout << "Threads: " << match.getNumThreads() << std::endl;
        std::cout << "MaxGames: " << match.getMaxGames() << std::endl;

        match.start();

    }

    if (vm.count("generate")) {

        std::string engine = vm["generate"].as<std::string>();
        std::string path = "Engines/" + engine;

        Engine myEngine(path);
        Generator generator(myEngine, "Positions/genBook6.pos", "TrainData/ultra3.game");
        generator.setThreads(95);
        generator.setMaxGames(10000000);
        generator.setTime(50);
        generator.start();

    }

    /*   std::ofstream stream;
       stream.open("TrainData/compressed.game");
       boost::iostreams::filtering_streambuf<boost::iostreams::output> out;
       out.push(boost::iostreams::bzip2_compressor());
       out.push(stream);
   */


    initialize();
    Weights<double> myWeights;
    WorkerPool pool(100, 1, myWeights);

    Value data[pool.batchSize];
    pool.setOutput(data);

    std::vector<Position> positions;

    Utilities::loadPositions(positions, "Positions/3move.pos");

    std::cout << "Positions: " << positions.size() << "\n";


    for (int i = 0; i < pool.batchSize; ++i) {
        pool.addWork(positions[i]);
    }
    pool.startThreads();
    pool.waitAll();

    for(int i=0;i<100;++i){
        std::cout<<pool.results[i]<<"\n";
    }







    /*   Engine engineOne("Engines/exp.so");
    Engine engineTwo("Engines/normal.so");
    engineOne.initialize();
    engineTwo.initialize();

    engineOne.setHashSize(24);
    engineTwo.setHashSize(24);

    std::vector<Position> positions;
    Utilities::loadPositions(positions,"Positions/3move.pos");
    Score result =Utilities::playGame(engineOne,engineTwo,positions[8],3000,true);





*/





/*

    using namespace Training;

    initialize();
    std::vector<TrainingGame> data;
    loadGames(data, "TrainData/test6comp.game");

    std::cout << "Length: " << data.size() << std::endl;
*/



/*
 std::vector<TrainingGame>remPos;
  std::vector<TrainingGame>data;


  Training::loadGames(data,"TrainData/ultra3.game");
    Training::loadGames(data,"TrainData/test5comp.game");


    SimplePool pool(95,data,remPos,0.90);
    pool.waitAll();
    std::cout<<"Removed: "<<remPos.size()<<std::endl;

    saveGames(remPos,"TrainData/test6comp.game");
    std::cout<<"Saved Data: "<<remPos.size()<<std::endl;*/






/*
    TrainingData trainData;

    for (TrainingGame &game : data) {
        if(game.result!=INVALID)
            game.extract(trainData);
    }

    trainData = TrainingData(trainData, [](TrainingPos pos) {

        if(_popcnt32(pos.pos.BP|pos.pos.WP)<=4)
            return false;

        MoveListe liste;
        getMoves(pos.pos, liste);
        Board board;
        BoardFactory::setUpPosition(board, pos.pos);
        Line localPV;
        Value testValue = quiescene(board, -INFINITE, INFINITE,localPV, 0);
        if (testValue.isWinning())
            return false;


        return true;
    });




    std::cout << "Starting " << std::endl;
    Weights<double> myWeights;

    Trainer trainer(myWeights, trainData);

    trainer.setLearningRate(1);
    trainer.setEpochs(1000);
    trainer.setl2Reg(0.000001);
    trainer.setCValue(-0.01);

    double loss = trainer.calculateLoss();
    std::cout << "Loss before : " << loss << std::endl;

    trainer.startTune();*/

}