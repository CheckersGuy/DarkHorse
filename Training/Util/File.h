//
// Created by robin on 18.12.21.
//

#ifndef READING_FILE_H
#define READING_FILE_H

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include "fcntl.h"
#include <string>
#include <fstream>
#include <iterator>
#include <Sample.h>
#include <BloomFilter.h>
#include <string>
#include <memory>
#include <filesystem>

namespace File {

    bool file_equal(std::string file_one,std::string file_two);



}
#endif //READING_FILE_H
