//
// Created by robin on 06.10.21.
//

#ifndef READING_POSSTREAMER_H
#define READING_POSSTREAMER_H

#include "Util/SampleUtil.h"
#include "generator.pb.h"
#include <../CheckerEngineX/Position.h>
#include <Sample.h>
#include <chrono>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <sys/mman.h>

class PosStreamer {

private:
  size_t gen_seed;
  std::string file_path;
  size_t buffer_size;
  std::vector<Sample> buffer;
  std::vector<Proto::Game> data;
  size_t ptr;
  std::ifstream stream;
  std::mt19937_64 generator;
  bool shuffle{true};
  bool is_raw_data{false};
  size_t num_samples{0}; // number of samples
  size_t offset{0};
  // in case we have a 'raw'file
  Sample *mapped;
  int fd;

public:
  PosStreamer(std::string file_path, size_t buff_size = 200000,
              size_t seed = 12312312) {

    this->file_path = file_path;
    gen_seed = seed;
    buffer_size = buff_size;
    generator = std::mt19937_64(seed);
    if (file_path.empty()) {
      std::cerr << "An empty path was given" << std::endl;
      std::exit(-1);
    }
    if (file_path.ends_with(".raw")) {
      is_raw_data = true;
    }
    // loading the game
    if (!is_raw_data) {
      stream = std::ifstream(file_path, std::ios::binary);
      if (!stream.good()) {
        std::cerr << "Could not open the stream" << std::endl;
        std::cerr << "FileName: " << file_path << std::endl;
        std::exit(-1);
      }

      Proto::Batch batch;
      batch.ParseFromIstream(&stream);
      std::cout << "Counting number of positions" << std::endl;
      size_t size = 0;
      for (auto game : batch.games()) {
        data.emplace_back(game);
      }
      for (Proto::Game game : data) {
        size += game.move_indices_size() + 1;
      }
      std::cout << "Counted: " << size << " positions" << std::endl;
      num_samples = size;
    } else {
      // we memory map the entire data !!!
      struct stat s;
      fd = open(file_path.c_str(), O_RDONLY);
      if (fd == -1) {
        std::cerr << "Could not open file" << std::endl;
        std::exit(-1);
      }
      auto r = fstat(fd, &s);
      num_samples = s.st_size / sizeof(Sample);
      mapped = (Sample *)mmap(0, s.st_size, PROT_READ, MAP_SHARED, fd, 0);
      buffer_size = std::min(num_samples, buffer_size);
      ptr = buffer.size() + 1000;
    }
  }
  ~PosStreamer() {
    if (is_raw_data) {
      munmap(mapped, num_samples * sizeof(Sample));
    }
  }

  Sample get_next();

  void set_shuffle(bool shuff);

  size_t get_buffer_size() const;

  size_t ptr_position();

  size_t get_num_positions() const;

  const std::string &get_file_path();
};

#endif // READING_POSSTREAMER_H
