
//
#include <chrono>
#include <assert.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>
#include "../Sample.h"
#include "../../Checkers/CheckerEngineX/MGenerator.h"
#include "SampleUtil.h"

std::vector<Sample> extract_sample(const Proto::Game& game){
  //extracting samples;
  std::vector<Sample> samples;
  Position current;
  current = Position::pos_from_fen(game.start_position());
  
  Sample first;
  first.position = current;
  samples.emplace_back(current);

  for(const auto& index : game.move_indices()){
    MoveListe liste;
    get_moves(current, liste);
    current.make_move(liste[index]);
    Sample s;
    s.position = current;
    samples.emplace_back(s);

  }

  //getting the game result
  MoveListe endlist;
  get_moves(current, endlist);
  Result end_result =UNKNOWN;
  if(endlist.length() ==0){
    end_result =((current.get_color() == BLACK)?WHITE_WON : BLACK_WON);
  }
  Sample last = samples.back();
  auto count = std::count(samples.begin(),samples.end(),last);
  if(count>=3){
    end_result = DRAW;
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
  struct stat s;
  int status;
  Sample * mapped;
  fd = open(raw_data.c_str(),O_RDONLY);
  status = fstat(fd,&s);
  std::cout<<"size: "<<s.st_size/sizeof(Sample)<<std::endl;
  
  mapped = (Sample*)mmap(0,s.st_size,PROT_READ,MAP_SHARED,fd,0);

  for(auto i=0;i<10;++i){
    Sample current = mapped[i];
    current.position.print_position();
    std::cout<<std::endl;
  }


  close(fd);

}
