
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
  MoveListe liste;
  std::vector<Sample> samples;
  Board board;
  board = Position::pos_from_fen(game.start_position());
  for (const auto &index : game.move_indices()) {
    liste.reset();
    get_moves(board.get_position(), liste);
    Move move = liste[index];
    Sample s;
    s.position = board.get_position();
    if (!move.is_capture()) {
      s.move = Statistics::MovePicker::get_policy_encoding(
          s.position.get_color(), liste[index]);
    }
    samples.emplace_back(s);
    board.make_move(move);
  }
  // getting the game result

  liste.reset();
  get_moves(board.get_position(), liste);
  Result end_result = DRAW;
  if (liste.length() == 0) {
    end_result = ((board.get_mover() == BLACK) ? WHITE_WON : BLACK_WON);
  }

  if (game.move_indices_size() >= 500) {
    end_result = UNKNOWN;
  }

  for (Sample &sample : samples) {
    sample.result = (end_result);
  }

  return samples;
}

void write_raw_data(std::string input_proto) {
  // Note: Use ftruncate to set the size of the file
  auto output_name = input_proto + ".raw";
  Sample *mapped;
  Proto::Batch batch;
  std::ifstream stream(input_proto);
  batch.ParseFromIstream(&stream);
  size_t counter = 0;
  size_t total_counter = 0;
  for (auto game : batch.games()) {
    auto samples = extract_sample(game);
    for (auto s : samples) {
      total_counter++;
      if (!s.is_training_sample()) {
        continue;
      }
      counter++;
    }
  }
  std::cout << "Counted training samples" << std::endl;
  std::cout << "ValidSamples: " << counter << std::endl;
  std::cout << "TotalSamples: " << total_counter << std::endl;
  std::mt19937_64 generator(getSystemTime());
  FILE *fp = fopen(output_name.c_str(), "w");
  ftruncate(fileno(fp), sizeof(Sample) * counter);
  fclose(fp);
  int fd = open(output_name.c_str(), O_RDWR);

  mapped = (Sample *)mmap(0, sizeof(Sample) * counter, PROT_WRITE | PROT_READ,
                          MAP_SHARED, fd, 0);
  const size_t counted = counter;
  counter = 0;
  for (auto game : batch.games()) {
    auto samples = extract_sample(game);
    for (auto s : samples) {
      if (!s.is_training_sample()) {
        continue;
      }
      if (((counter + 1) % 100000) == 0) {
        double perc = (double)counter;
        perc /= (double)counted;
        std::cout << "Progress: " << perc << std::endl;
      }
      mapped[counter++] = s;
    }
  }

  for (auto i = 0; i < 1000; ++i) {
    mapped[i].position.print_position();
    std::cout << std::endl;
  }

  munmap(mapped, counted * sizeof(Sample));
}

void sort_raw_data(std::string raw_data, std::string copy) {
  std::mt19937_64 generator(3123123131ull);
  struct stat s;
  int status;
  std::vector<uint32_t> indices;
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

  std::sort(samples.begin(), samples.end(), [](Sample one, Sample two) {
    auto key1 = Zobrist::generate_key(one.position);
    auto key2 = Zobrist::generate_key(two.position);
    return key1 < key2;
  });
  std::vector<Sample> reduced_samples;
  int i;
  for (i = 0; i < num_samples; ++i) {
    int start;
    bool changed = false;
    // scanning through the file
    Position start_pos = samples[i].position;
    for (start = i; start < num_samples; ++start) {
      auto current = samples[start];
      if (current.position != start_pos) {
        break;
      }
      if (current.result != samples[i].result) {
        changed = true;
      }
    }
    if (changed) {
      for (auto k = i; k < start; ++k) {
        reduced_samples.emplace_back(samples[k]);
      }
    } else {
      reduced_samples.emplace_back(samples[i]);
    }
    i = start;
  }
  std::shuffle(reduced_samples.begin(), reduced_samples.end(), generator);

  std::ofstream out_stream(copy, std::ios::binary);
  if (!out_stream.good()) {
    std::cerr << "Error" << std::endl;
    std::exit(-1);
  }

  out_stream.write((char *)&reduced_samples[0],
                   sizeof(Sample) * (reduced_samples.size()));
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
  batch.ParseFromIstream(&stream);
  for (auto game : batch.games()) {
    auto samples = extract_sample(game);
    int end_cutoff = -1;
    for (int k = 0; k < samples.size(); ++k) {
      auto sample = samples[k];
      if (!filter.has(sample.position)) {
        stats.num_unqiue++;
        filter.insert(sample.position);
        stats.bucket_distrib[sample.position.bucket_index()]++;
        auto piece_count = sample.position.piece_count();
        if (piece_count <= 10) {
          end_cutoff = k;
        }
      }
    }
    if (end_cutoff != -1) {
      stats.endgame_cutoff_ply += end_cutoff;
    }

    stats.num_positions += samples.size();
    auto result = get_game_result(game);
    stats.num_wins += (result != DRAW);
    stats.num_draws += (result == DRAW);
  }
  stats.endgame_cutoff_ply /= stats.num_unqiue;
}
