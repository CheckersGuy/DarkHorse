#include "Node.h"
#include <cmath>

bool Node::is_leaf() { return child_nodes.empty(); }

Node::Node(Move move, double prob) {
  this->move = move;
  this->prob = prob;
}

bool Node::expand(Position pos) {

  MoveListe liste;
  get_moves(pos, liste);
  if (liste.length() == 0) {
    return true;
  }
  // for now just a uniform transform
  const double prob = 1.0 / static_cast<double>(liste.length());

  for (auto move : liste) {
    child_nodes.emplace_back(std::make_unique<Node>(move, prob));
  }

  if (liste.length() == 1 || liste[0].is_capture()) {
    return false;
  }

  int32_t *out = policy.get_raw_eval(pos);
  double total = 0.0;
  for (int i = 0; i < liste.length(); ++i) {
    auto move = liste[i];
    if (pos.color == BLACK) {
      move = move.flipped();
    }
    auto encoding = move.get_move_encoding();
    assert(encoding >= 0 && encoding < 128);
    auto logit = out[encoding];
    double logit_f = static_cast<double>(logit) / 128.0;

    double temp = std::exp(logit_f);
    assert(i < child_nodes.size());
    child_nodes[i]->prob = temp;
    total += temp;
  }
  for (int i = 0; i < child_nodes.size(); ++i) {
    child_nodes[i]->prob /= total;
  }

  return false;
}
Node *Node::select_best_child() {

  Node *max_child = child_nodes[0].get();
  for (auto &node : child_nodes) {
    if (node->visits > max_child->visits) {
      max_child = node.get();
    }
  }
  assert(max_child != nullptr);
  return max_child;
}

double Node::q_value() {
  const double f_visits = Node::eps + (double)visits;
  return value / f_visits;
}

double Node::uct(Node *parent) {
  assert(parent != nullptr);
  const double f_child_visits = (double)visits;
  const double f_parent_visits = (double)parent->visits;

  const double cpuct = 1.0 * 0.74; // some guessing here
  const double explore =
      cpuct * prob * (std::sqrt(std::log(f_parent_visits) / f_child_visits));

  return -q_value() + explore;
}

Node *Node::select_best_uct(Board &board) {
  Node *max_child = nullptr;
  double max_value = -10000000.0;
  for (auto &node : child_nodes) {
    if (node->visits == 0) {
      return node.get();
    }

    auto value = node->uct(this);
    board.make_move(node->move);
    if (board.is_repetition()) {
      value = -100000.0;
    }
    board.undo_move();
    if (value > max_value) {
      max_child = node.get();
      max_value = value;
    }
  }
  assert(max_child != nullptr);
  return max_child;
}

Node *Node::select_best_prior() {
  Node *max_child = nullptr;
  double max_value = -10000000.0;
  for (auto &node : child_nodes) {

    const auto value = node->prob;
    if (value > max_value) {
      max_child = node.get();
      max_value = value;
    }
  }
  assert(max_child != nullptr);
  return max_child;
}
