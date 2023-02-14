//
// Created by robin on 06.10.21.
//

#include <sys/stat.h>
#include "PosStreamer.h"
#include "Sample.h"

size_t PosStreamer::get_num_positions() const {
    return num_samples;
}

Sample PosStreamer::get_next() {
      if (ptr >= buffer.size()) {
            buffer.clear();
            //Need to fill the buffer again;
            std::cout<<"Filling up the buffer"<<std::endl;
            std::cout<<"Buffersize: "<<buffer_size<<std::endl;
            while(buffer.size()<buffer_size){
              if(!is_raw_data){
              auto game = data[offset++];
              if(offset>=data.size()){
                offset =0;
                std::shuffle(data.begin(),data.end(),generator);
              }
              auto positions = extract_sample(game);
              for(auto pos : positions){
                buffer.emplace_back(pos);
              }
            }
            else{
              buffer.emplace_back(mapped[offset++]);
              offset=offset%num_samples;
            }

             if (shuffle) {
            std::cout<<"Shuffled"<<std::endl;
            auto t1 = std::chrono::high_resolution_clock::now();
            std::shuffle(buffer.begin(), buffer.end(), generator);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto dur = t2-t1;
        }
        ptr =0;
    }
  }
    Sample next = buffer[ptr++];
    std::string result_string;
    /*
    if(next.result ==DRAW){
      result_string="DRAW";
    }else if(next.result == WHITE_WON)
      result_string ="WHITE_WON";
    else if(next.result ==BLACK_WON)
      result_string="BLACK_WON";
    else
      result_string ="UNDEFINED";
    std::cout<<result_string<<std::endl;
    */
    return next;
} 



size_t PosStreamer::get_buffer_size() const {
    return buffer_size;
}

size_t PosStreamer::ptr_position() {
    return ptr;
}

const std::string &PosStreamer::get_file_path() {
    return file_path;
}


size_t PosStreamer::get_file_size() const {
    std::filesystem::path my_path(file_path);
    return std::filesystem::file_size(my_path);
}

void PosStreamer::set_shuffle(bool shuff) {
    shuffle=shuff;
}

