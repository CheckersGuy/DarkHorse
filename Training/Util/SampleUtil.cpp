
//
#include "SampleUtil.h"
#include "../../Checkers/CheckerEngineX/MGenerator.h"
#include "../Sample.h"
#include <algorithm>
#include <assert.h>
#include <chrono>
#include <cstdint>
#include <fcntl.h>
#include <fstream>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <thread>
#include <vector>

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
      s.position.key = Zobrist::generate_key(s.position);
      train_samples.emplace_back(s);
    }
  }
  std::cout << "NumSamples: " << train_samples.size() << std::endl;
  out_stream.write((char *)&train_samples[0],
                   sizeof(Sample) * train_samples.size());
}

void sort_raw_data(std::string raw_data, std::string copy) {
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
  std::shuffle(samples.begin(), samples.end(), generator);

  out_stream.write((char *)&samples[0], sizeof(Sample) * (samples.size()));
}
// needs to be redone
void count_real_duplicates(std::string raw_data, std::string output) {
  std::ofstream stream(output, std::ios::binary);
  if (!stream.good()) {
    std::exit(-1);
  }
  int fd; // file-descriptor
  size_t size;
  struct stat s;
  int status;
  Sample *mapped;
  fd = open(raw_data.c_str(), O_RDWR);
  status = fstat(fd, &s);
  size = s.st_size;
  std::cout << "size: " << s.st_size / sizeof(Sample) << std::endl;

  mapped = (Sample *)mmap(0, size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
  auto num_samples = size / sizeof(Sample);

  std::sort(mapped, mapped + num_samples,
            [&](const Sample &one, const Sample &two) {
              auto key1 = Zobrist::generate_key(one.position);
              auto key2 = Zobrist::generate_key(two.position);
              return key1 > key2;
            });
  size_t counter = 0;
  for (auto i = 0; i < num_samples;) {
    auto sample = mapped[i];
    size_t start = i;
    while (sample.position == mapped[i].position) {
      ++i;
    }
    bool switched = false;
    for (auto k = start; k < i; ++k) {
      if (mapped[k].result != mapped[start].result) {
        switched = true;
        break;
      }
    }
    if (switched) {
      for (auto k = start; k < i; ++k) {
        stream << mapped[k];
        counter++;
      }
    } else {
      counter++;
      stream << sample;
    }
    if ((counter % 1000) == 0) {
      std::cout << "Counter: " << counter << std::endl;
    }
  }
  std::cout << "New size : " << counter << " vs old_size : " << num_samples
            << std::endl;
  munmap(mapped, size);

  close(fd);
  stream = std::ofstream("debug.txt");
}

void create_shuffled_raw(std::string input_prot) { write_raw_data(input_prot); }

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
  Zobrist::init_zobrist_keys();
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
      stats.bucket_distrib[sample.position.bucket_index()]++;
    }
    auto result = get_game_result(game);
    stats.num_wins += (result != DRAW);
    stats.num_draws += (result == DRAW);
  }
  stats.num_positions = counter;
}
