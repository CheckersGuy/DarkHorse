#ifndef NODE
#define NODE
#include "GameLogic.h"
#include "MGenerator.h"
#include <iostream>
class Node {
  static constexpr double eps = 0.0000001;

public:
  double value = 0.0;
  double prob;
  uint32_t visits = 0;
  Move move; // is empty for the root node
  std::vector<std::unique_ptr<Node>> child_nodes;
  bool is_terminal = false;

public:
  Node(Move move, double prob);
  bool is_leaf(); // returns true if the node has not been expanded yet;
  bool expand(Position pos); // creates child nodes

  Node *
  select_best_child(); // selects the best child according to the uct-formula
  Node *select_best_uct(Board &board);
  Node *select_best_prior();
  double q_value();
  double uct(Node *parent, Board &board);
};

#endif
