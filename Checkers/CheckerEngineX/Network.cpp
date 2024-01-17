//
// Created by root on 18.04.21.
//

#include "Network.h"
#include "Bits.h"
#include "GameLogic.h"
#include "types.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

Value tempo_white(Position pos) {
  Value score = 0;
  for (auto i = 0; i < 7; ++i) {
    score += (i + 1) * Bits::pop_count(MASK_ROWS[i] & (pos.WP & (~pos.K)));
  }
  return score;
}
Value tempo_black(Position pos) {
  Value score = 0;
  for (auto i = 7; i >= 1; --i) {
    score += (8 - i) * Bits::pop_count(MASK_ROWS[i] & (pos.BP & (~pos.K)));
  }
  return score;
}

Value win_eval(TB_RESULT result, Value score, Position pos) {
  // helper to encourage finishing the game
  auto BK = pos.BP & pos.K;
  auto WK = pos.WP & pos.K;
  auto total_pieces = pos.piece_count();
  auto eval = 100 * (Bits::pop_count(pos.WP & (~pos.K)) -
                     Bits::pop_count(pos.BP & (~pos.K)));
  eval +=
      140 * (Bits::pop_count(pos.WP & pos.K) - Bits::pop_count(pos.BP & pos.K));

  eval += (Bits::pop_count(pos.WP) -
           Bits::pop_count(pos.BP) * (240 - total_pieces * 20));

  eval += (Bits::pop_count(WK & CENTER) - Bits::pop_count(BK & CENTER)) * 8;
  // penalty for having kings in a single corner
  eval += (Bits::pop_count(WK & SINGLE_CORNER) -
           Bits::pop_count(BK & SINGLE_CORNER)) *
          -25;

  if (result == TB_RESULT::WIN) {
    if (pos.color == WHITE) {
      eval += 5 * tempo_white(pos);
    } else {
      eval -= 5 * tempo_black(pos);
    }
  }

  return score + eval * pos.color;
}

void Accumulator::refresh() {
  for (auto i = 0; i < OutDim; ++i) {
    white_acc[i] = ft_biases[i];
    black_acc[i] = ft_biases[i];
  }
  previous_black = Position{};
  previous_white = Position{};
}

void Accumulator::load_weights(std::ifstream &stream) {
  ft_weights = (int16_t *)std_aligned_alloc(Network::ALIGNMENT,
                                            (120 * OutDim) * sizeof(int16_t));
  ft_biases = (int16_t *)std_aligned_alloc(Network::ALIGNMENT,
                                           OutDim * sizeof(int16_t));
  stream.read((char *)ft_weights, sizeof(int16_t) * (OutDim * 120));
  stream.read((char *)ft_biases, sizeof(int16_t) * (OutDim));

  for (auto i = 0; i < OutDim; ++i) {
    black_acc[i] = ft_biases[i];
    white_acc[i] = ft_biases[i];
  }
}

Accumulator::~Accumulator() {
  std_aligned_free(ft_biases);
  std_aligned_free(ft_weights);
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

  auto WP_Z =
      (~after.get_pieces<WHITE, PAWN>()) & (before.get_pieces<WHITE, PAWN>());
  auto BP_Z =
      (~after.get_pieces<BLACK, PAWN>()) & (before.get_pieces<BLACK, PAWN>());
  auto WK_Z =
      (~after.get_pieces<WHITE, KING>()) & (before.get_pieces<WHITE, KING>());
  auto BK_Z =
      (~after.get_pieces<BLACK, KING>()) & (before.get_pieces<BLACK, KING>());

  int offset = 0;
  int num_active = 0;
  int num_removed = 0;
  while (WP_O) {
    auto index = Bits::bitscan_foward(WP_O) - 4 + offset;
    active_features[num_active++] = index;
    WP_O &= WP_O - 1;
  }
  offset += 28;

  while (BP_O) {
    auto index = Bits::bitscan_foward(BP_O) + offset;
    active_features[num_active++] = index;
    BP_O &= BP_O - 1;
  }
  offset += 28;

  while (WK_O) {
    auto index = Bits::bitscan_foward(WK_O) + offset;
    active_features[num_active++] = index;
    WK_O &= WK_O - 1;
  }
  offset += 32;

  while (BK_O) {
    auto index = Bits::bitscan_foward(BK_O) + offset;
    active_features[num_active++] = index;
    BK_O &= BK_O - 1;
  }

  offset = 0;

  while (WP_Z) {
    auto index = Bits::bitscan_foward(WP_Z) - 4 + offset;
    removed_features[num_removed++] = index;
    WP_Z &= WP_Z - 1;
  }
  offset += 28;

  while (BP_Z) {
    auto index = Bits::bitscan_foward(BP_Z) + offset;
    removed_features[num_removed++] = index;
    BP_Z &= BP_Z - 1;
  }
  offset += 28;

  while (WK_Z) {
    auto index = Bits::bitscan_foward(WK_Z) + offset;
    removed_features[num_removed++] = index;
    WK_Z &= WK_Z - 1;
  }
  offset += 32;

  while (BK_Z) {
    auto index = Bits::bitscan_foward(BK_Z) + offset;
    removed_features[num_removed++] = index;
    BK_Z &= BK_Z - 1;
  }

  auto *accu = reinterpret_cast<__m256i *>(input);
  constexpr int num_regs = 16; // number of available avx2 registers
  constexpr int OutRegisters = OutDim / 16; // each register can hold 16 int16_t
  constexpr int num_chunks =
      OutRegisters / num_regs; // we have 16 avx2 registers

  for (auto k = 0; k < num_chunks; ++k) {
    __m256i regs[num_regs];

    for (auto i = 0; i < num_regs; ++i) {
      regs[i] = _mm256_load_si256(accu + i + k * num_regs);
    }
    for (auto i = 0; i < num_active; ++i) {
      const __m256i *weights =
          reinterpret_cast<__m256i *>(ft_weights + OutDim * active_features[i]);

      for (auto j = 0; j < num_regs; ++j) {
        regs[j] = _mm256_add_epi16(
            _mm256_load_si256(weights + j + k * num_regs), regs[j]);
      }
    }

    for (auto i = 0; i < num_removed; ++i) {
      const __m256i *weights = reinterpret_cast<const __m256i *>(
          ft_weights + OutDim * removed_features[i]);
      for (auto j = 0; j < num_regs; ++j) {
        regs[j] = _mm256_sub_epi16(
            regs[j], _mm256_load_si256(weights + j + k * num_regs));
      }
    }
    for (auto i = 0; i < num_regs; ++i) {
      _mm256_store_si256(accu + i + k * num_regs, regs[i]);
    }
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

void Accumulator::update(Position after) {
  update(BLACK, after);
  update(WHITE, after);
}

uint8_t *Accumulator::forward(uint8_t *in, const Position &next) {
  int16_t *z_previous;
  if (next.color == BLACK) {
    z_previous = black_acc;
  } else {
    z_previous = white_acc;
  }
  update(next.color, next);
  psqt = z_previous[OutDim - 1];
  Simd::accum_activation8<OutDim>(z_previous, in);

  return in;
}

void Network::load_bucket(std::string file) {

  std::ifstream stream(file, std::ios::binary);
  if (!stream.good()) {
    std::cerr << "Could not load network file, path " << file << std::endl;
    std::exit(-1);
  }
  // need to load buckets
  accumulator.load_weights(stream);
  first.load_params(stream);
  second.load_params(stream);
  output.load_params(stream);
}

int32_t *Network::compute_incre_forward_pass(Position next) {
  auto bucket_index = next.bucket_index();
  auto *out = accumulator.forward(input, next);

  out = first.forward(out, bucket_index);
  out = second.forward(out, bucket_index);
  return output.forward(out, bucket_index);
}

int Network::operator[](int index) { return input[index]; }

int Network::evaluate(Position pos, int ply, int shuffle) {

  if (pos.BP == 0 && pos.color == BLACK) {
    return loss(ply);
  }

  if (pos.WP == 0 && pos.color == WHITE) {
    return loss(ply);
  }
  auto result = tablebase.probe(pos);
  if (result != TB_RESULT::UNKNOWN) {
    auto tb_value = (result == TB_RESULT::WIN)    ? TB_WIN
                    : (result == TB_RESULT::LOSS) ? TB_LOSS
                                                  : 0;
    // return tb_value;
    return win_eval(result, tb_value, pos);
  }
  // trying some heuristic for endgames

  const auto nnue = *compute_incre_forward_pass(pos);
  auto eval = (nnue);
  return eval;
}

int Network::get_raw_eval(Position pos) {

  const auto nnue = *compute_incre_forward_pass(pos);
  auto eval = (nnue);
  return eval;
}
