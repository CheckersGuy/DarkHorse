//
// Created by robin on 06.10.21.
//

#ifndef READING_POSSTREAMER_H
#define READING_POSSTREAMER_H

#include <string>
#include <fcntl.h>
#include <memory>
#include <../CheckerEngineX/Position.h>
#include <fstream>
#include <Sample.h>
#include <iterator>
#include <filesystem>
#include "Util/Compress.h"
#include <chrono>
#include "generator.pb.h"
#include "Util/SampleUtil.h"
#include <string>
#include <stdio.h>
#include <stdlib.h>
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
    size_t num_samples; // number of samples
    size_t offset{0};
    //in case we have a 'raw'file
    Sample * mapped;
    size_t file_size;
    int fd;
   

public:

    PosStreamer(std::string file_path, size_t buff_size=200000, size_t seed = 12312312){
      
                
        this->file_path = file_path;
        gen_seed=seed;
        buffer_size = buff_size;
        generator = std::mt19937_64(getSystemTime());
        if (file_path.empty()) {
            std::cerr << "An empty path was given" << std::endl;
            std::exit(-1);
        }
            if(file_path.ends_with(".raw")){
          is_raw_data = true;
        }
        //loading the game
        if(!is_raw_data){
        stream = std::ifstream(file_path, std::ios::binary);
        if (!stream.good()) {
            std::cerr << "Could not open the stream" << std::endl;
            std::cerr << "FileName: " << file_path << std::endl;
            std::exit(-1);
        }

        Proto::Batch batch;
        batch.ParseFromIstream(&stream);
        std::cout<<"Counting number of positions"<<std::endl; 
        size_t size =0;
        for(auto game : batch.games()){
          data.emplace_back(game);
        }
        for(Proto::Game game : data){
          size+=game.move_indices_size()+1;
        }
        std::cout<<"Counted: "<<size<<" positions"<<std::endl;
        num_samples=size;
        }else{
          //we memory map the entire data !!!
          struct stat s;
          fd = open(file_path.c_str(),O_RDWR);
          auto r =fstat(fd,&s);
          file_size = s.st_size;
          num_samples = file_size/sizeof(Sample);
          mapped = (Sample*)mmap(0,file_size,PROT_READ |PROT_WRITE,MAP_SHARED,fd,0);
        }
        ptr = buffer.size()+1000;

    }
    ~PosStreamer(){
      if(is_raw_data){
        munmap(mapped,file_size);
      }
    }

    Sample get_next();

    void set_shuffle(bool shuff);

    size_t get_buffer_size() const;

    size_t ptr_position();

    size_t get_file_size() const;

    size_t get_num_positions() const;

    const std::string &get_file_path();



};

#endif //READING_POSSTREAMER_H
