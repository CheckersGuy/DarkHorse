#include <initializer_list>
#include <iostream>
#include "../generator.pb.h"
#include "../Sample.h"
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>

struct GameStat{
  size_t num_wins{0},num_draws{0};
  size_t num_positions{0};

}

std::vector<Sample> extract_sample(const Proto::Game& game);

void write_raw_data(std::string input_proto);

void sort_raw_data(std::string raw_data);

void create_shuffled_raw(std::string input_proto);

void view_game(std::string input_proto, int index);

void get_game_stats(std::string input_proto,GameStat& stats);

Proto::Result get_game_result(Proto::Game game);
