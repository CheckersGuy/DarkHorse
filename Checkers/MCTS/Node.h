#ifndef NODE
#define NODE
#include "MGenerator.h"
#include <iostream>
class Node {

  float value = 0.0;
  uint32_t visits = 0;
  Move move; // is empty for the root node
  std::vector<Node> child_nodes;

public:
  Node(Move move);
  bool is_leaf(); // returns true if the node has not been expanded yet;
  void expand(Position pos); // creates child nodes

  Node *
  select_best_chid(); // selects the best child according to the uct-formula
};

#endif
