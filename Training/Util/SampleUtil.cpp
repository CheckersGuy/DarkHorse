
//
#include "SampleUtil.h"
#include "../../Checkers/CheckerEngineX/MGenerator.h"
#include "../Sample.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <vector>

void gather_positions(std::vector<std::string> data, std::string output) {

  // works for game_data and sample data
  std::ofstream out_stream(output);
  if (!out_stream.good()) {
    std::exit(-1);
  }

  BloomFilter<Position> filter(40711816683, 6);
  size_t unique_counter = 0;
  auto from_samples = [&](std::string input) {
    std::ifstream stream(input);
    if (!stream.good()) {
      std::exit(-1);
    }

    for (;;) {
      Sample current;
      stream.read((char *)&current, sizeof(Sample));
      if (!filter.has(current.position)) {
        unique_counter++;
        out_stream.write((char *)&current.position, sizeof(Sample));
      }
      filter.insert(current.position);
      if (!stream.good()) {
        break;
      }
    }
  };

  for (auto &file : data) {
    from_samples(file);
  }
  std::cout << "UniquePositions: " << unique_counter << std::endl;
}

std::vector<Sample> extract_sample(const Proto::Game &game) {
  // extracting samples;
  std::vector<Sample> samples;
  Board board;
  board = Position::pos_from_fen(game.start_position());
  for (const auto &index : game.move_indices()) {

    MoveListe liste;
    get_moves(board.get_position(), liste);
    Move move = liste[index];
    Sample s;
    s.position = board.get_position();
    if (!move.is_capture()) {
      s.move = Statistics::MovePicker::get_policy_encoding(
          s.position.get_color(), liste[index]);
    }
    samples.emplace_back(s);
    board.play_move(move);
  }
  // getting the game result
  MoveListe liste;
  get_moves(board.get_position(), liste);
  Result end_result = DRAW;
  if (liste.length() == 0) {
    end_result = ((board.get_mover() == BLACK) ? WHITE_WON : BLACK_WON);
  }

  if (game.move_indices_size() >= 600) {
    end_result = UNKNOWN;
  }

  for (Sample &sample : samples) {
    sample.result = (end_result);
  }

  return samples;
}

void write_raw_data(std::string input_proto) {
  // Note: Use ftruncate to set the size of the file
  std::string output_name = input_proto + ".raw";
  Proto::Batch batch;
  std::ifstream stream(input_proto);
  if (!stream.good()) {
    std::exit(-1);
  }
  std::ofstream out_stream(output_name);
  if (!out_stream.good()) {
    std::exit(-1);
  }
  batch.ParseFromIstream(&stream);
  size_t counter = 0;
  size_t total_counter = 0;
  std::vector<Sample> train_samples;
  for (auto game : batch.games()) {
    auto samples = extract_sample(game);
    for (auto s : samples) {
      total_counter++;
      if (!s.is_training_sample()) {
        continue;
      }
      train_samples.emplace_back(s);
    }
  }
  std::mt19937_64 generator(32112415ull);
  std::shuffle(train_samples.begin(), train_samples.end(), generator);
  std::cout << "NumSamples: " << train_samples.size() << std::endl;
  out_stream.write((char *)&train_samples[0],
                   sizeof(Sample) * train_samples.size());
}

void sort_raw_data(std::string raw_data, std::string copy) {
  struct Entry {
    int w_win{0};
    int b_win{0};
    int draw{0};
    int unknown{0};
  };
  std::mt19937_64 generator(3123123131ull);
  struct stat s;
  int status;
  int fd = open(raw_data.c_str(), O_RDWR);
  if (fd == -1) {
    std::cout << "Error" << std::endl;
    std::cout << copy << std::endl;
    std::exit(-1);
  }
  status = fstat(fd, &s);
  close(fd);
  auto num_samples = s.st_size / sizeof(Sample);

  std::vector<Sample> samples;
  for (auto i = 0; i < num_samples; ++i) {
    samples.emplace_back(Sample{});
  }
  std::ifstream stream(raw_data, std::ios::binary);
  if (!stream.good()) {
    std::cerr << "Error" << std::endl;
    std::exit(-1);
  }
  stream.read((char *)&samples[0], sizeof(Sample) * num_samples);
  std::ofstream out_stream(copy, std::ios::binary);
  if (!out_stream.good()) {
    std::cerr << "Error" << std::endl;
    std::exit(-1);
  }

  // inserting into hashtable
  std::unordered_map<Position, Entry> my_map;

  for (auto &sample : samples) {
    if (sample.result == UNKNOWN || !sample.is_training_sample())
      continue;
    Entry &entry = my_map[sample.position];
    entry.w_win += (sample.result == WHITE_WON);
    entry.b_win += (sample.result == BLACK_WON);
  }
  samples.clear();
  std::vector<Sample> new_samples;
  size_t second_counter = 0;
  for (auto &[key, value] : my_map) {
    auto count = (value.b_win != 0) + (value.w_win != 0) + (value.draw != 0);

    // Case only wins,draws
    if (count == 1) {
      Sample next;
      next.position = key;
      if (value.b_win != 0) {
        next.result = BLACK_WON;
      } else if (value.w_win != 0) {
        next.result = WHITE_WON;
      } else if (value.draw != 0) {
        next.result = DRAW;
      } else {
        next.result = UNKNOWN;
      }
      if (next.result != UNKNOWN) {
        new_samples.emplace_back(next);
      }
    } else {
      second_counter++;
      if ((second_counter % 10000) == 0) {
        std::cout << "NumSecondCases: " << second_counter << std::endl;
      }
      for (auto i = 0; i < value.b_win; ++i) {
        Sample next;
        next.position = key;
        next.result = BLACK_WON;
        new_samples.emplace_back(next);
      }
      for (auto i = 0; i < value.w_win; ++i) {
        Sample next;
        next.position = key;
        next.result = WHITE_WON;
        new_samples.emplace_back(next);
      }
      for (auto i = 0; i < value.draw; ++i) {
        Sample next;
        next.position = key;
        next.result = DRAW;
        new_samples.emplace_back(next);
      }
    }
  }
  std::shuffle(new_samples.begin(), new_samples.end(), generator);
  out_stream.write((char *)&new_samples[0],
                   sizeof(Sample) * new_samples.size());
}

void view_game(std::string input_proto, int index) {
  Proto::Batch batch;
  std::ifstream stream(input_proto);
  if (!stream.good()) {
    std::cerr << "Could not open stream" << std::endl;
    std::exit(-1);
  }
  batch.ParseFromIstream(&stream);
  auto game = batch.games(index);
  std::vector<Sample> samples = extract_sample(game);
  for (auto sample : samples) {
    sample.position.print_position();
  }
}

Result get_game_result(Proto::Game game) {
  MoveListe liste;
  auto samples = extract_sample(game);
  auto last = samples.back();
  Result result = last.result;
  return result;
}
void get_game_stats(std::string input_proto, GameStat &stats) {
  BloomFilter<Position> filter(9585058378, 7);
  Proto::Batch batch;
  std::ifstream stream(input_proto);
  if (!stream.good()) {
    std::cerr << "Could not open stream" << std::endl;
    std::exit(-1);
  }

  size_t counter = 0;
  batch.ParseFromIstream(&stream);
  for (auto game : batch.games()) {
    auto samples = extract_sample(game);
    int end_cutoff = -1;
    for (int k = 0; k < samples.size(); ++k) {
      auto sample = samples[k];
      if (!sample.is_training_sample())
        continue;
      counter++;
      if (!filter.has(sample.position)) {
        stats.num_unqiue++;
        filter.insert(sample.position);
      }
    }
    auto result = get_game_result(game);
    stats.num_wins += (result != DRAW);
    stats.num_draws += (result == DRAW);
  }
  stats.num_positions = counter;
}
