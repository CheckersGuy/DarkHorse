#include <iostream>
#include "GameLogic.h"
#include "Position.h"
#include "types.h"
#include <regex>
#include <algorithm>

using SelfGame = std::pair<std::string,std::vector<uint8_t>>;

inline std::vector<std::string>split_string(std::string input, char delim){

  std::vector<std::string>output;
  std::string word;
  for(auto i=0;i<input.length();++i){
    if(input[i]==delim){
      output.emplace_back(word);
      word = "";
    }else{
      word+=input[i];
    }
  }
  if(!word.empty())
    output.emplace_back(word);
  return output;
}


class Selfplay{
  private:
  int time_per_move{100};
  int hash_size{21};
  int adjud{10};
  bool stop{false};
  
  public:

  void start_loop();

  SelfGame play_game(std::string fen_string);

  void send_game(SelfGame& game);

  void terminate();

  void set_hash_size(int hash);

  void set_time_per_move(int time);

  void set_adjudication(int adj);

  void parse_command(std::string command);
  


};
