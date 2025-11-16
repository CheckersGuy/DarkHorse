#include "Search.h"

Value evaluate_board(Board &board, int ply) {
  if (!board.get_position().is_legal()) {
    board.print_board();
  }
  assert(board.get_position().is_legal());
  if (board.is_repetition()) {
    return 0;
  }
  return evaluate(board.get_position(), ply);
  Value bestValue = -INFINITE;
  if (board.get_position().is_end()) {
    return loss(ply);
  }
  MoveListe moves;
  get_captures(board.get_position(), moves);

  if (moves.length() == 0) {
    return evaluate_board(board, ply);
  }
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

double value_to_float(Value value) {
  const double temp = static_cast<double>(value) / 128.0;
  return 2.0 * sigmoid(temp) - 1.0;
}

void MCTSSearch::simulate(Board board) {
  std::vector<Node *> visited;
  simul_count++;
  Node *current = root.get();
  assert(current != nullptr);
  current->visits++;
  while (!current->is_leaf()) {
    // node should have at least 1 child
    Node *next = current->select_best_uct();
    assert(next != nullptr);
    board.make_move(next->move);
    visited.emplace_back(next);

    current = next;
    if (current->is_terminal) {
      break;
    }
  }

  // expanding the node
  const bool is_loss = current->expand(board.get_position());
  // handling terminal states
  if (!is_loss) {
    current = current->select_best_prior();
    visited.emplace_back(current);
    board.make_move(current->move);
  } else if (is_loss) {
    current->is_terminal = true;
    current->value = -1.0;
  }
  assert(current != nullptr);
  double backup_value;
  if (current->is_terminal) {
    backup_value = current->value;
  } else {
    backup_value = value_to_float(evaluate_board(board, visited.size()));
  }

  for (int i = visited.size() - 1; i >= 0; i--) {
    visited[i]->visits++;
    visited[i]->value += backup_value;
    backup_value = -backup_value;
    board.undo_move();
  }
  root->value += backup_value;
}

Move MCTSSearch::search(Board board) {

  MoveListe liste;
  get_moves(board.get_position(), liste);
  if (liste.length() == 1) {
    return liste[0];
  }

  auto start = std::chrono::high_resolution_clock::now();

  while (true) {
    auto end = std::chrono::high_resolution_clock::now();
    auto dur = end - start;
    auto time_taken =
        std::chrono::duration_cast<std::chrono::milliseconds>(dur);
    simulate(board);

    if (time_taken.count() >= max_time) {
      break;
    }
  }
  std::cout << root->select_best_child()->move << std::endl;
  return Move{};
}

std::vector<Move> MCTSSearch::get_pv() {
  std::vector<Move> moves;

  Node *current = root.get();

  while (!current->is_leaf()) {
    Node *next = current->select_best_child();
    moves.emplace_back(next->move);
    current = next;
  }

  return moves;
}
