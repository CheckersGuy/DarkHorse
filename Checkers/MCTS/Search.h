#include "GameLogic.h"
#include "MCTSUtil.h"
#include "Node.h"
#include "types.h"
#include <cmath>
#include <memory>
#include <random>
// basically the same as qs but without threat-extensions

Value evaluate_board(Board &board, int ply);
double value_to_float(Value value);
Value float_to_value(double q_value);
class MCTSSearch {

public:
  std::unique_ptr<Node> root;
  uint32_t simul_count = 0;
  uint32_t max_nodes = 1000;
  size_t max_time = 30000;
  std::mt19937_64 generator; // better do something else to seed the generator

public:
  MCTSSearch() : generator(12312321ull) {
    root = std::make_unique<Node>(Move{}, 1.0);
  };
  void simulate(Board board);
  Move search(Board board);
  std::vector<Move> get_pv();
};
