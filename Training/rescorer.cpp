#include "Sample.h"
#include <fstream>
#include <iostream>
#include <iterator>
int main(int argl, const char **argc) {
  std::cout << "Hello I am the rescorer" << std::endl;

  std::string file_name = argc[1];

  std::ifstream stream("../Training/TrainData/" + file_name);

  std::istream_iterator<Sample> begin(stream);
  std::istream_iterator<Sample> end{};

  std::for_each(begin, end, [](Sample s) {
    s.position.print_position();
    std::cout << std::endl;
  });
}
