#include "MGenerator.h"
#include "Position.h"
#include <iostream>

int main(int argl, const char **argc) {

  std::cout << "Das wird meine MCTS-Engine" << std::endl;

  const auto pos = Position::get_start_position();
  pos.print_position();
  return 0;
}
