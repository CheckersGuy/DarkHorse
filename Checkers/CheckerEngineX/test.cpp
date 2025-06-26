#include "Bits.h"
#include "GameLogic.h"
#include "Network.h"
#include "Position.h"
#include "Simd.h"
#include "incbin.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <emmintrin.h>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <random>
// finding nonzero values very quickly

INCBIN(mlh_net, "mlh3.quant");
INCBIN(network, "shuffled_21.quant");
INCBIN(policy, "policybigger3.quant");
int main() {
  // generate random numbers and measure
  mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
  network.load_from_array(gnetworkData, gnetworkSize);
  policy.load_from_array(gpolicyData, gpolicySize);
  const auto start_pos = Position::get_start_position();
  start_pos.print_position();
  TT.resize_in_mb(128);
  Board board = Board(start_pos);
  Move best;
  const int depth = 25;
  const auto time = 30000;
  const auto max_nodes = 1000000000ull;
  auto value =
      searchValue(board, best, depth, time, max_nodes, true, std::cout);

  auto stats = network.accumulator.get_activation_stats();

  std::cout << "NumNNZ: " << stats.first
            << " and NumNNZ_blocks: " << stats.second << std::endl;

#ifdef SPARSEOPT
  std::cout << "Testing something" << std::endl;
#endif
  return 0;
}
