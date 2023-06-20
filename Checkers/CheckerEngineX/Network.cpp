//
// Created by root on 18.04.21.
//

#include "Network.h"
#include "GameLogic.h"
#include "types.h"
#include <cstdint>
#include <cstdlib>

// Note:: Should look at the accumulator because it's using the input variable
// as well

Layer::Layer(int in, int out, int8_t *w, int32_t *b)
    : in_features(in), out_features(out), weights(w), bias(b) {}

void Accumulator::refresh() {
  for (auto i = 0; i < OutDim; ++i) {
    white_acc[i] = ft_biases[i];
    black_acc[i] = ft_biases[i];
  }
  previous_black = Position{};
  previous_white = Position{};
}

void Accumulator::load_weights(std::ifstream &stream) {
  ft_weights = (int16_t *)std::aligned_alloc(Network::ALIGNMENT,
                                             (120 * OutDim) * sizeof(int16_t));
  ft_biases = (int16_t *)std::aligned_alloc(Network::ALIGNMENT,
                                            OutDim * sizeof(int16_t));
  stream.read((char *)ft_weights, sizeof(int16_t) * (OutDim * 120));
  stream.read((char *)ft_biases, sizeof(int16_t) * (OutDim));

  for (auto i = 0; i < OutDim; ++i) {
    black_acc[i] = ft_biases[i];
    white_acc[i] = ft_biases[i];
  }
}

Accumulator::~Accumulator() {
  free(ft_biases);
  free(ft_weights);
}

void Accumulator::apply(Color perp, Position before, Position after) {
  int16_t *input = ((perp == BLACK) ? black_acc : white_acc);

  auto WP_O =
      after.get_pieces<WHITE, PAWN>() & (~before.get_pieces<WHITE, PAWN>());
  auto BP_O =
      after.get_pieces<BLACK, PAWN>() & (~before.get_pieces<BLACK, PAWN>());
  auto WK_O =
      after.get_pieces<WHITE, KING>() & (~before.get_pieces<WHITE, KING>());
  auto BK_O =
      after.get_pieces<BLACK, KING>() & (~before.get_pieces<BLACK, KING>());

  size_t offset = 0;

  while (WP_O) {
    auto index = Bits::bitscan_foward(WP_O) - 4 + offset;
    add_feature(input, index);
    WP_O &= WP_O - 1;
  }
  offset += 28;

  while (BP_O) {
    auto index = Bits::bitscan_foward(BP_O) + offset;
    add_feature(input, index);
    BP_O &= BP_O - 1;
  }
  offset += 28;

  while (WK_O) {
    auto index = Bits::bitscan_foward(WK_O) + offset;
    add_feature(input, index);
    WK_O &= WK_O - 1;
  }
  offset += 32;

  while (BK_O) {
    auto index = Bits::bitscan_foward(BK_O) + offset;
    add_feature(input, index);
    BK_O &= BK_O - 1;
  }

  auto WP_Z =
      (~after.get_pieces<WHITE, PAWN>()) & (before.get_pieces<WHITE, PAWN>());
  auto BP_Z =
      (~after.get_pieces<BLACK, PAWN>()) & (before.get_pieces<BLACK, PAWN>());
  auto WK_Z =
      (~after.get_pieces<WHITE, KING>()) & (before.get_pieces<WHITE, KING>());
  auto BK_Z =
      (~after.get_pieces<BLACK, KING>()) & (before.get_pieces<BLACK, KING>());

  offset = 0;

  // to be continued
  while (WP_Z) {
    auto index = Bits::bitscan_foward(WP_Z) - 4 + offset;
    remove_feature(input, index);
    WP_Z &= WP_Z - 1;
  }
  offset += 28;

  while (BP_Z) {
    auto index = Bits::bitscan_foward(BP_Z) + offset;
    remove_feature(input, index);
    BP_Z &= BP_Z - 1;
  }
  offset += 28;

  while (WK_Z) {
    auto index = Bits::bitscan_foward(WK_Z) + offset;
    remove_feature(input, index);
    WK_Z &= WK_Z - 1;
  }
  offset += 32;

  while (BK_Z) {
    auto index = Bits::bitscan_foward(BK_Z) + offset;
    remove_feature(input, index);
    BK_Z &= BK_Z - 1;
  }
}

void Accumulator::update(Color perp, Position after) {
  if (perp == BLACK) {
    apply(perp, previous_black.get_color_flip(), after.get_color_flip());
    previous_black = after;
  } else {
    apply(perp, previous_white, after);
    previous_white = after;
  }
}

void Network::load_bucket(std::string file) {

  std::ifstream stream(file, std::ios::binary);
  if (!stream.good()) {
    std::cerr << "Could not load network file, path " << file << std::endl;
    std::exit(-1);
  }
  accumulator.load_weights(stream);
  first.load_params(stream);
  second.load_params(stream);
  output.load_params(stream);
}

int32_t *Network::compute_incre_forward_pass(Position next) {
  int16_t *z_previous;
  if (next.color == BLACK) {
    z_previous = accumulator.black_acc;
  } else {
    z_previous = accumulator.white_acc;
  }
  accumulator.update(next.color, next);
  Simd::accum_activation8<2048>(z_previous, input);
  auto *out = first.forward(input);
  out = second.forward(out);
  auto *eval = output.forward(out);
  return eval;
}

int Network::operator[](int index) { return input[index]; }

int Network::evaluate(Position pos, int ply) {

  if (pos.BP == 0 && pos.get_color() == BLACK) {
    return -loss(ply);
  }
  if (pos.WP == 0 && pos.get_color() == WHITE) {
    return loss(ply);
  }
  return *compute_incre_forward_pass(pos);
}
