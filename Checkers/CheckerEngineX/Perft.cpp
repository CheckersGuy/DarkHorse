
//
// Created by Robin on 14.01.2018.
//

#include "Perft.h"
#include "Bits.h"

namespace Perft {

// only counting the non-capturing moves to see
// if the rest implementation does anything wrong there
//

uint64_t perft_check(Position pos, int depth) {
  MoveListe liste;
  get_moves(pos, liste);
  if (depth == 0) {
    return 1;
  }
  uint64_t counter = 0;

  for (int i = 0; i < liste.length(); ++i) {
    Position copy = pos;
    copy.make_move(liste[i]);
    if (depth == 1) {
      std::cout << "----------------------" << std::endl;
      pos.print_position();
      copy.print_position();
    }
    counter += perft_check(copy, depth - 1);
  }

  return counter;
}

} // namespace Perft
