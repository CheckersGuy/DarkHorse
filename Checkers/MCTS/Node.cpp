#include "Node.h"

bool Node::is_leaf() { return child_nodes.empty(); }

Node::Node(Move move) { this->move = move; }

void Node::expand(Position pos) {
  MoveListe liste;
  get_moves(pos, liste);

  for (auto move : liste) {
    child_nodes.emplace_back(Node(move));
  }
}

Node *Node::select_best_chid() {
  // TODO
}
