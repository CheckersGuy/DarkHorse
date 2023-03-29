
//
#include <chrono>
#include <assert.h>
#include <fstream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <vector>
#include "../Sample.h"
#include "../../Checkers/CheckerEngineX/MGenerator.h"
#include "SampleUtil.h"



std::vector<Sample> extract_sample(const Proto::Game& game){
  //extracting samples;
  MoveListe liste;
  std::vector<Sample> samples;
  Board board;
  board = Position::pos_from_fen(game.start_position());
  for(const auto& index : game.move_indices()){
    liste.reset();
    get_moves(board.get_position(), liste);
    Move move = liste[index];
    Sample s;
    s.position = board.get_position();
    if(!move.is_capture()){
      s.move = Statistics::MovePicker::get_policy_encoding(s.position.get_color(), liste[index]);
    } 
    samples.emplace_back(s);
    board.make_move(move);
  }
  //getting the game result
  
  liste.reset();
  get_moves(board.get_position(), liste);
  Result end_result =DRAW;
  if(liste.length() ==0){
    end_result =((board.get_mover() == BLACK)?WHITE_WON : BLACK_WON);
  }
  
  if(game.move_indices_size()>=500){
    end_result = UNKNOWN;
  }

  for(Sample& sample : samples){
    sample.result = (end_result);
  }

  return samples;


}


void write_raw_data(std::string input_proto){
  auto output_name = input_proto+".raw";
  std::ifstream stream(input_proto);
  if(!stream.good()){
    std::cerr<<"Could not load the file"<<std::endl;
    std::cerr<<"File: "<<input_proto<<std::endl;
    std::exit(-1);
  }
  std::ofstream out_stream(output_name);
  Proto::Batch batch;
  batch.ParseFromIstream(&stream);
  for(auto game : batch.games()){
    auto samples = extract_sample(game);
    for(auto s : samples){
      out_stream<<s;
    }
  }

}


void sort_raw_data(std::string raw_data){
  int fd; // file-descriptor
  size_t size;
  struct stat s;
  int status;
  Sample * mapped;
  fd = open(raw_data.c_str(),O_RDWR);
  status = fstat(fd,&s);
  size = s.st_size;
  std::cout<<"size: "<<s.st_size/sizeof(Sample)<<std::endl;
  
  mapped = (Sample*)mmap(0,size,PROT_READ |PROT_WRITE,MAP_SHARED,fd,0);
  auto num_samples = size/sizeof(Sample);
  for(auto i=0;i<500;++i){
    Sample current = mapped[i];
    current.position.print_position();
    std::cout<<std::endl;
  }
  Zobrist::init_zobrist_keys();
  std::hash<Sample> hasher;
  std::mt19937_64 generator;
  std::uniform_int_distribution<size_t> distrib;
  
  std::sort(mapped,mapped+num_samples,[&](const Sample& one,const  Sample& two){
        return distrib(generator)> distrib(generator);
      });
      
  munmap(mapped, size);

  close(fd);

}


void create_shuffled_raw(std::string input_prot){
 write_raw_data(input_prot);
 sort_raw_data(input_prot+".raw");
}



void view_game(std::string input_proto,int index){
  Proto::Batch batch;
  std::ifstream stream(input_proto);
  if(!stream.good()){
    std::cerr<<"Could not open stream"<<std::endl;
    std::exit(-1);
  }
  batch.ParseFromIstream(&stream);
  auto game = batch.games(index);
  std::vector<Sample>samples =  extract_sample(game);
  for(auto sample : samples){
    sample.position.print_position();
  }
}

Result get_game_result(Proto::Game game){
  MoveListe liste;
  auto samples = extract_sample(game);
  auto last = samples.back();
  Result result = last.result;
  return result;
}
void get_game_stats(std::string input_proto, GameStat &stats){
  Zobrist::init_zobrist_keys();
  BloomFilter<Position> filter(9585058378,7);
  Proto::Batch batch;
  std::ifstream stream(input_proto);
  if(!stream.good()){
    std::cerr<<"Could not open stream"<<std::endl;
    std::exit(-1);
  }
  batch.ParseFromIstream(&stream);
 for(auto game : batch.games()){
   auto samples = extract_sample(game);
   for(auto sample : samples){
     if(!filter.has(sample.position)){
        stats.num_unqiue++;
        filter.insert(sample.position);
     }
   }
   stats.num_positions+=samples.size();
   auto result = get_game_result(game);
   stats.num_wins+=(result !=DRAW);
   stats.num_draws+=(result == DRAW);
 }


}

