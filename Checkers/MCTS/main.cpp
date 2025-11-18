#include "MGenerator.h"
#include "Node.h"
#include "Position.h"
#include "Search.h"
#include "incbin.h"
#include "types.h"
#include <chrono>
#include <iostream>
INCBIN(mlh_net, "mlh3.quant");
INCBIN(network, "registry_128.quant");
INCBIN(policy, "policybigger6.quant");

INCBIN(mlh_perm, "mlh.perm");
INCBIN(net_perm, "evalpermutation.perm");

// printing the probs of all moves in a position

void print_probs(Position pos) {
  int32_t *out = &policy.out_layer.buffer[0];
  MoveListe liste;
  get_moves(pos, liste);
  out = policy.get_raw_eval(pos);
  double total = 0.0;
  std::vector<double> softmax_values;
  for (auto move : liste) {
    Move copy = move;
    if (pos.color == BLACK) {
      move = move.flipped();
    }
    auto encoding = move.get_move_encoding();
    auto logit = out[encoding];
    double logit_f = static_cast<double>(logit) / 128.0;

    double temp = std::exp(logit_f);

    softmax_values.emplace_back(temp);
    total += temp;
  }
  for (auto i = 0; i < liste.length(); ++i) {

    softmax_values[i] /= total;
    std::cout << liste[i] << " " << softmax_values[i] << std::endl;
  }
}

int main(int argl, const char **argc) {
  mlh_net.load_permutation_from_array(gmlh_permData, gmlh_permSize);
  // policy.load_permutation_from_array(gpolicy_permDta, gpolicy_permSize);
  // network.load_permutation_from_array(gnet_permData, gnet_permSize);

  mlh_net.load_from_array(gmlh_netData, gmlh_netSize);
  network.load_from_array(gnetworkData, gnetworkSize);
  policy.load_from_array(gpolicyData, gpolicySize);
  const auto pos = Position::get_start_position();

  Board board = Board(pos);
  board.print_board();

  for (auto count = 0; count < 431; count++) {
    std::cout << "Startin search" << std::endl;
    MCTSSearch search = MCTSSearch();
    if (board.get_mover() == BLACK) {
      search.max_time = 1500;
    } else {
      search.max_time = 1;
    }

    auto best_move = search.search(board);

    std::cout << "RootValue: " << search.root->q_value() << std::endl;
    std::cout << "Count: " << search.simul_count << std::endl;
    std::cout << "BestMove: " << best_move << std::endl;
    std::cout << "FenString:" << board.get_position().get_fen_string()
              << std::endl;

    std::cout.flush();
    board.play_move(best_move);
    board.print_board();

    MoveListe liste;
    get_moves(board.get_position(), liste);
    if (liste.length() == 0) {
      break;
    }
  }

  return 0;
}
