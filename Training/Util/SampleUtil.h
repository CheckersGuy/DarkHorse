#include "../BloomFilter.h"
#include "../Sample.h"
#include "../generator.pb.h"
#include <fcntl.h>
#include <initializer_list>
#include <iostream>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>

// endgame_cutoff_ply is the length of a game ignoring all positions with less
// then <=10 pieces

struct GameStat {
  size_t num_wins{0}, num_draws{0};
  size_t num_positions{0};
  size_t num_unqiue{0};
  double endgame_cutoff_ply{0};
  friend std::ostream &operator<<(std::ostream &stream, const GameStat &other) {
    stream << "Wins: " << other.num_wins << " Draws: " << other.num_draws
           << " NumPositions: " << other.num_positions
           << " Unique: " << other.num_unqiue << std::endl;
    return stream;
  }
};

void count_real_duplicates(std::string raw_data, std::string output);

std::vector<Sample> extract_sample(const Proto::Game &game);

void write_raw_data(std::string input_proto);

void sort_raw_data(std::string raw_data, std::string copy);

void view_game(std::string input_proto, int index);

void get_game_stats(std::string input_proto, GameStat &stats);
