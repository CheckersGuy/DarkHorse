#include "Search.h"

Value evaluate_board(Board &board, int ply) {

  Value bestValue = -INFINITE;

  if (board.is_silent_position(board.get_mover())) {
    if (board.get_position().is_end()) {
      return loss(ply);
    }
    return evaluate(board.get_position(), ply);
  }
  MoveListe moves;
  get_captures(board.get_position(), moves);

  for (int i = 0; i < moves.length(); ++i) {

    Move move = moves[i];
    board.make_move(move);
    Value value;
    value = -evaluate_board(board, ply + 1);
    board.undo_move();

    if (value > bestValue) {
      bestValue = value;
    }
  }

  return bestValue;
}
