//
// Created by leagu on 13.09.2021.
//
#include <chrono>
#include <assert.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "egdb.h"
#include <thread>
#include <vector>
#include "MGenerator.h"
#include "../../Training/Sample.h"
#include "../../Training/BloomFilter.h"
#include "../../Training/Util/Compress.h"
#include "CmdParser.h"

#define DB_PATH "D:\kr_english_wld"



void print_msgs(char *msg) {
    printf("%s", msg);
}


Result get_tb_result(Position pos, int max_pieces, EGDB_DRIVER *handle) {
    if (pos.has_jumps() || Bits::pop_count(pos.BP | pos.WP)>max_pieces)
        return UNKNOWN;


    EGDB_NORMAL_BITBOARD board;
    board.white = pos.WP;
    board.black = pos.BP;
    board.king = pos.K;

    EGDB_BITBOARD normal;
    normal.normal = board;
    auto val = handle->lookup(handle, &normal, ((pos.color == BLACK) ? EGDB_BLACK : EGDB_WHITE), 0);

    if (val == EGDB_UNKNOWN)
        return UNKNOWN;

    if (val == EGDB_WIN)
        return (pos.color == BLACK) ? BLACK_WON : WHITE_WON;

    if (val == EGDB_LOSS)
        return (pos.color == BLACK) ? WHITE_WON : BLACK_WON;

    if (val == EGDB_DRAW)
        return DRAW;


    return UNKNOWN;
}
void create_samples_from_games(std::string games, std::string output, int max_pieces, EGDB_DRIVER *handle,int num_threads) {

    //speeding things up with some more threads
    std::ifstream stream(games,std::ios::binary);
    if(!stream.good()) {
        std::cerr<<"Could not open input stream"<<std::endl;
        std::exit(-1);
    }
    std::ofstream out_stream(output, std::ios::binary);
    if(!out_stream.good()) {
        std::cerr<<"Could not open output stream"<<std::endl;
        std::exit(-1);
    };

//loaind the unrescored games
//
    std::vector<Game>unrescored;
    std::istream_iterator<Game>begin(stream);
    std::istream_iterator<Game>end;

    std::copy(begin,end,std::back_inserter(unrescored));

    const auto num_games = unrescored.size();
    const auto num_chunks = num_threads;
    const auto chunk_size = num_games/num_chunks;
    const auto left_overs = num_games-chunk_size*num_chunks;

    std::vector<std::thread>threads;
	auto oracle =[&](Position pos){
			return get_tb_result(pos,max_pieces,handle);
	};
    for(auto i=0; i<num_chunks; ++i) {

        threads.emplace_back(std::thread([&]() {
            auto lower = i*chunk_size;
            auto upper =lower+chunk_size;
            upper = std::min(upper,num_games);
            for(auto k=lower; k<upper; ++k) {
                auto& game = unrescored[k];
				game.rescore_game(oracle);
            }

        }));
		//rescoring the leftovers
    }

	for(auto i=num_chunks*chunk_size;i<unrescored.size();++i){
				unrescored[i].rescore_game(oracle);
	}

	for(auto& th : threads){
			th.join();
	}
}
void create_samples_from_games(std::string input_file, std::string output, int max_pieces, EGDB_DRIVER *handle) {

    std::ifstream stream(input_file,std::ios::binary);
    if(!stream.good()) {
        std::cerr<<"Could not open input stream"<<std::endl;
        std::cerr<<input_file<<std::endl;
        std::exit(-1);
    }
    std::ofstream out_stream(output, std::ios::binary);
    if(!out_stream.good()) {
        std::cerr<<"Could not open output stream"<<std::endl;
        std::exit(-1);
    }
    std::istream_iterator<Game> begin(stream);
    std::istream_iterator<Game>end;
    std::vector<Game>games;
    std::copy(begin,end,std::back_inserter(games));


    std::cout<<"NumGames: "<<games.size()<<std::endl;
	

	auto oracle = [&](Position pos){
		return get_tb_result(pos,max_pieces,handle);
	};
	
    const auto num_games_perc = games.size()/20;
    size_t counter=0;
	auto start = std::chrono::high_resolution_clock::now();
	
    std::for_each(games.begin(), games.end(), [&](Game& game) {
            game.rescore_game(oracle);
            counter++;
			if(counter%1000 == 0){
				std::cout<<"Counter: "<<counter<<std::endl;
			}

    });
    //write back in bulk
    std::cout<<"Writing the data"<<std::endl;
    std::copy(games.begin(),games.end(),std::ostream_iterator<Game>(out_stream));
}


int main(int argl, const char **argc) {

    int i, status, max_pieces, nerrors;
    EGDB_TYPE egdb_type;
    EGDB_DRIVER *handle;

    /* Check that db files are present, get db type and size. */
    status = egdb_identify(DB_PATH, &egdb_type, &max_pieces);
    std::cout<<"MAX_PIECES: "<<max_pieces<<std::endl;

    if (status) {
        printf("No database found at %s\n", DB_PATH);
        return (1);
    }
    printf("Database type %d found with max pieces %d\n", egdb_type, max_pieces);

    /* Open database for probing. */
    handle = egdb_open(EGDB_NORMAL, max_pieces, 4000, DB_PATH, print_msgs);
    if (!handle) {
        printf("Error returned from egdb_open()\n");
        return (1);
    }
    std::cout<<"Starting Rescoring the training data"<<std::endl;
    std::string in_file("../Training/TrainData/reinf.train");
    std::string out_file("../Training/TrainData/reinfformatted.train");

    create_samples_from_games(in_file, out_file, max_pieces, handle);
    std::cout<<"Done rescoring"<<std::endl;
    handle->close(handle);

    return 0;
}
