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
INCBIN(network, "buckets1024_107.quant");
INCBIN(policy, "policybigger6.quant");
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
  const auto time = 3000;
  const auto max_nodes = 1000000000ull;

  auto root_moves =
      searchValueMultiPV(board, 1, MAX_PLY, time, max_nodes, false, std::cout);
  auto stats = network.accumulator.get_activation_stats();

  std::cout << "NumNNZ: " << stats.first
            << " and NumNNZ_blocks: " << stats.second << std::endl;

  return 0;
}
