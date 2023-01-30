
//
#include <chrono>
#include <assert.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "egdb.h"
#include <thread>
#include <vector>
#include "../../Checkers/CheckerEngineX/MGenerator.h"
#define DB_PATH "D:\kr_english_wld"
#include "SampleUtil.h"

std::vector<Proto::Sample> extract_sample(Proto::Game& game, int max_pieces,EGDB_DRIVER* handle){
  //extracting samples;
  std::vector<Proto::Sample> samples;
  Position current;
  current = Position::pos_from_fen(game.start_position());
  Proto::Sample first;
  first.set_mover((current.get_color() == BLACK) ? Proto::BLACK : Proto::WHITE);
  first.set_wp(current.WP);
  first.set_bp(current.BP);
  first.set_k(current.K);

  for(const auto& index : game.move_indices()){
    MoveListe liste;
    get_moves(current, liste);
    current.make_move(liste[index]);
    Proto::Sample sample;
    sample.set_bp(current.BP);
    sample.set_wp(current.WP);
    sample.set_k(current.K);
    sample.set_mover((current.get_color() == BLACK)? Proto::BLACK : Proto::WHITE);
    samples.emplace_back(sample);
  }

  //getting the game result
  MoveListe endlist;
  get_moves(current, endlist);
  auto end_result =Proto::DRAW;
  if(endlist.length() ==0){
    end_result =(current.get_color() == BLACK)?Proto::WHITE_WIN : Proto::BLACK_WIN;
  }
  for(Proto::Sample& sample : samples){
    sample.set_result(end_result);
  }
  int last_stop =-1;
  for(auto i =0;i<samples.size();++i){
    auto result =get_tb_result(samples[i],max_pieces,handle);
    if(result !=Proto::UNDEFINED){
      for(auto k=i;k>last_stop;k--){
       samples[k].set_result(result);
    }
      last_stop = i;
    }
    }

  //Now we can do the rescoring

}




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

Proto::Result get_tb_result(Proto::Sample sample,int max_pieces,EGDB_DRIVER* handle){
  Position temp;
  temp.BP = sample.bp();
  temp.WP = sample.wp();
  temp.K = sample.k();
  temp.color = (sample.mover() == Proto::WHITE) ? WHITE : BLACK;
  auto result =  get_tb_result(pos,max_pieces,handle);
  if(result == WHITE_WON){
     return Proto::WHITE_WIN;
  }else if(result == BLACK_WON){
    return Proto::Black_WIN;
  }else if(result == DRAW){
    return Proto::DRAW;
  }
  return Proto::UNDEFINED

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
    Proto::Batch batch;
    Proto::TrainData out_data;

    std::cout<<"NumGames: "<<batch.games_size()<<std::endl;
	

	auto oracle = [&](Position pos){
		return get_tb_result(pos,max_pieces,handle);
	};
	
    size_t counter=0;
	auto start = std::chrono::high_resolution_clock::now();
	
  for(auto& game : batch.games){
    auto rescored_samples = extract_sample(game);
    for(auto& sample : rescored_samples){
      out_data.add_samples(sample);
      counter++;
      if((counter%1000) ==0){
        std::cout<<"Counter: "<<counter<<std::endl;
      }
    }
  }
  out_data.SerializeToOstream(&out_stream);

}



