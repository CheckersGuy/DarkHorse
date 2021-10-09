//
// Created by robin on 06.10.21.
//

#include <sys/stat.h>
#include "PosStreamer.h"


Sample PosStreamer::get_next() {
    if (ptr >= buffer_size) {
        ptr = 0;
        //if we reached the end of our file
        //we have to wrap around
        size_t read_elements = 0;
        do {
            if (stream.peek() == EOF) {
                stream.clear();
                stream.seekg(std::ios::beg);
            }
            std::istream_iterator<Sample> begin(stream);
            std::istream_iterator<Sample> end;
            for (; (begin != end) && read_elements < buffer_size; ++begin) {
                buffer[read_elements++] = (*begin);
            }
        } while (read_elements < buffer_size);
        //std::cout << "buffer is full now" << std::endl;
        //the buffer is filled now so we can shuffle the elements
        if (shuffle) {
            std::shuffle(buffer.get(), buffer.get() + buffer_size, generator);
            //std::cout << "buffer is shuffled" << std::endl;
        }
    }
    return buffer[ptr++];
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

size_t PosStreamer::get_file_size(std::string path) {
    struct stat stat_buf;
    int rc = stat(path.c_str(), &stat_buf);
    return rc == 0 ? stat_buf.st_size : -1;
}

size_t PosStreamer::get_file_size() const {
    return file_size;
}
