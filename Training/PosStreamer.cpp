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
                if(shuffle){
                  std::cout<<"Reached end of training games"<<std::endl;
                }
                offset =0;
                std::shuffle(data.begin(),data.end(),generator);
              }
              auto positions = extract_sample(game);
              for(auto pos : positions) {
                if(!pos.is_training_sample())
                  continue;
                buffer.emplace_back(pos);
              }
            }
            else{
              Sample s = mapped[offset++];
              if(s.is_training_sample()){
                buffer.emplace_back(s);        
              }
              size_t samples_in_file =file_size/sizeof(Sample);
              if(offset>=samples_in_file){
                offset=0;
              };
            }
            }
             if (shuffle) {
            std::cout<<"Shuffled"<<std::endl;
            std::cout<<"Offset: "<<offset<<std::endl;
            std::cout<<"BufferSize after fill: "<<buffer.size()<<std::endl;
            auto t1 = std::chrono::high_resolution_clock::now();
            std::shuffle(buffer.begin(), buffer.end(), generator);
            auto t2 = std::chrono::high_resolution_clock::now();
            auto dur = t2-t1;
        }
        ptr =0;
    
  }
    Sample next = buffer[ptr++];
    
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

void PosStreamer::set_shuffle(bool shuff) {
    shuffle=shuff;
}

