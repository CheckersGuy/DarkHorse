
//
// Created by Robin on 14.01.2018.
//

#include "Perft.h"

namespace Perft {

uint64_t perft_check(Position pos, int depth) {
  MoveListe liste;
  get_moves(pos, liste);
  if (depth == 1) {
    return liste.length();
  }
  uint64_t counter = 0;

  for (int i = 0; i < liste.length(); ++i) {
    Position copy = pos;
    copy.make_move(liste[i]);
    counter += perft_check(copy, depth - 1);
  }

  return counter;
}

} // namespace Perft
