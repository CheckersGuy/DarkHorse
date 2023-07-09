#include "Sample.h"
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argl, const char **argc) {
  std::cout << "Hello I am the rescorer" << std::endl;

  std::string file_name = argc[1];
  std::string path = "../Training/TrainData/" + file_name;
  std::cout << "Path: " << path << std::endl;
  std::ifstream stream(path.c_str());
  if (!stream.good()) {
    std::cout << "Could not open the stream" << std::endl;
  }

  /*
    std::istream_iterator<Sample> begin(stream);
    std::istream_iterator<Sample> end{};

    std::for_each(begin, end, [](Sample s) {
      s.position.print_position();
      std::cout << std::endl;
    });
    */
  Sample test;

  while (stream >> test) {
    std::cout << test << std::endl;
    std::cout << "----------------------------" << std::endl;
  }
}
