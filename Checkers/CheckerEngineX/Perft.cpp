
//
// Created by Robin on 14.01.2018.
//

#include "Perft.h"

namespace Perft {

uint64_t perft_check(Board &board, int depth) {
  MoveListe liste;
  get_moves(board.get_position(), liste);
  if (depth == 1) {
    return liste.length();
  }
  uint64_t counter = 0;

  for (int i = 0; i < liste.length(); ++i) {
    board.make_move(liste[i]);
    counter += perft_check(board, depth - 1);
    board.undo_move();
  }

  return counter;
}

} // namespace Perft
