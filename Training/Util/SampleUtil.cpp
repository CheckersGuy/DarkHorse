
//
#include <chrono>
#include <assert.h>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <thread>
#include <vector>
#include "../../Checkers/CheckerEngineX/MGenerator.h"
#include "SampleUtil.h"

std::vector<Proto::Sample> extract_sample(const Proto::Game& game){
  //extracting samples;
  std::vector<Proto::Sample> samples;
  Position current;
  current = Position::pos_from_fen(game.start_position());
  Proto::Sample first;
  first.set_mover((current.get_color() == BLACK) ? Proto::BLACK : Proto::WHITE);
  first.set_wp(current.WP);
  first.set_bp(current.BP);
  first.set_k(current.K);
  samples.emplace_back(first);

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
  std::string result_string;
  if(end_result ==Proto::DRAW)
    result_string="DRAW";
  else if(end_result ==Proto::BLACK_WIN)
    result_string="BLACK_WIN";
  else if(end_result ==Proto::WHITE_WIN)
    result_string="WHITE_WIN";
  //std::cout<<result_string<<std::endl;
  for(Proto::Sample& sample : samples){
    sample.set_result(end_result);
  }
  return samples;


}


