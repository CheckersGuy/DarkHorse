#include "Selfplay.h"
#include "Zobrist.h"
#include "types.h"
#include <unistd.h>
void Selfplay::start_loop(){
  //to be adde
  while(!stop){
    std::string line;
    std::getline(std::cin,line);
    parse_command(line);
  }
}

Selfplay::Selfplay(){
  int pid = getpid();
  Zobrist::init_zobrist_keys(pid^getSystemTime());
}


void Selfplay::set_time_per_move(int time){
  time_per_move = time;
}

void Selfplay::set_hash_size(int hash){
  hash_size = hash;
}

void Selfplay::set_adjudication(int adj){
 adjud =  adj;
}

void Selfplay::parse_command(std::string command){
  //parsing of commands
  if(command =="terminate"){
    stop =true;
  }


  auto split =split_string(command,'!');
  if(split.size()>1){ 
  if(split[0]=="playgame"){
    auto fen_string =split[1];
    auto game = play_game(fen_string);
    send_game(game);
  }
  //loading a new network file
  if(split[0] == "loadnetwork"){
    auto net_file = split[1];
    network.load(net_file);
  }

  if(split[0]=="settings"){
    auto time =std::stoi(split[1]);
    auto hash_size = std::stoi(split[2]);
    auto adj = std::stoi(split[3]);
    set_time_per_move(time);
    set_hash_size(hash_size);
    set_adjudication(adj);
    TT.resize(hash_size);
  }
  
  }
  std::cout<<std::flush;
 }


void Selfplay::send_game(SelfGame &game){
  std::cout<<"game_start"<<"\n";
  std::cout<<game.first<<"\n";
  for(auto& move_index : game.second){
    std::cout<<(int)move_index<<"\n";
  }
  std::cout<<"game_end"<<"\n";
  std::cout<<std::flush;
}

void Selfplay::terminate(){
  stop =true;
}

void Selfplay::set_resign_threshhold(float value){
  
}

SelfGame Selfplay::play_game(std::string fen_string){
//to be added
//testing resign threshholds next
  //network.load("Networks/client.quant");
  TT.clear();
  network.init();
  Statistics::mPicker.clear_scores();
  SelfGame game;
  Board board;
  board = Position::pos_from_fen(fen_string);
  game.first = fen_string;
  for(auto i=0;i<600;++i){
    Move best;
    MoveListe liste;
    get_moves(board.get_position(),liste);
    if(liste.length()==0){
      return game;
    }
    int k;
    if(liste.length() ==1){
      best = liste[0];
    }else{
    searchValue(board,best,MAX_PLY,time_per_move,false,std::cout);
    }
    for( k=0;k<liste.length();k++){
      if(liste[k] == best){
        break;
      }
    }
    game.second.emplace_back(k);
    board.make_move(best);
  
    bool is_rep = board.is_repetition2(0);
    if(is_rep)
      break;
  
  }

  return game;
}


