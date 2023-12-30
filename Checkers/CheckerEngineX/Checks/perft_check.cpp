
#include "../Perft.h"
#include <iostream>

int main(int argl, const char **argc) {
  size_t num_ply = std::stoi(argc[1]);

  constexpr std::array<uint64_t, 29> node_counts = {1ull,
                                                    7ull,
                                                    49ull,
                                                    302ull,
                                                    1469ull,
                                                    7361ull,
                                                    36768ull,
                                                    179740ull,
                                                    845931ull,
                                                    3963680ull,
                                                    18391564ull,
                                                    85242128ull,
                                                    388623673ull,
                                                    1766623630ull,
                                                    7978439499ull,
                                                    36263167175ull,
                                                    165629569428ull,
                                                    758818810990ull,
                                                    3493881706141ull,
                                                    16114043592799ull,
                                                    74545030871553ull,
                                                    345100524480819ull,
                                                    1602372721738102ull,
                                                    7437536860666213ull,
                                                    34651381875296000ull,
                                                    161067479882075800ull,
                                                    752172458688067137ull,
                                                    3499844183628002605ull,
                                                    16377718018836900735ull};

  // std::cout<<"NumNodes: "<<call_back.num_nodes<<std::endl;

  for (auto i = 1; i < std::min(num_ply + 1, node_counts.size()); ++i) {
    Board board;
    board = Position::get_start_position();

    std::cout << "Checking depth: " << i << " ";
    auto start_time = std::chrono::high_resolution_clock::now();
    auto count = Perft::perft_check(board, i);
    std::cout << count << std::endl;
    std::cout << count << " ";

    if (count != node_counts[i])
      return 1;
  }

  return 0;
}
