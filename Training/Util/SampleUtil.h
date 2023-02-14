#include <initializer_list>
#include <iostream>
#include "../generator.pb.h"
#include "../Sample.h"
#include <sys/stat.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <stdio.h>
std::vector<Sample> extract_sample(const Proto::Game& game);

void write_raw_data(std::string input_proto);

void sort_raw_data(std::string raw_data);
