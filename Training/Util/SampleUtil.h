#include <initializer_list>
#include <iostream>
#include "../generator.pb.h"
#include "../Sample.h"
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
#include "../BloomFilter.h"
struct GameStat{
  size_t num_wins{0},num_draws{0};
  size_t num_positions{0};
  size_t num_unqiue{0};
  friend std::ostream& operator <<(std::ostream&stream, const GameStat& other){
    stream<<"Wins: "<<other.num_wins<<" Draws: "<<other.num_draws<<" NumPositions: "<<other.num_positions<<" Unique: "<<other.num_unqiue;
    return stream;
  }
};

std::vector<Sample> extract_sample(const Proto::Game& game);

void write_raw_data(std::string input_proto);

void sort_raw_data(std::string raw_data);

void create_shuffled_raw(std::string input_proto);

void view_game(std::string input_proto, int index);

void get_game_stats(std::string input_proto,GameStat& stats);

Result get_game_result(Proto::Game game);
