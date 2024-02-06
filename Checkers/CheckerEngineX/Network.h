//
// Created by root on 18.04.21.
//

#ifndef READING_NETWORK_H
#define READING_NETWORK_H

#include "Bits.h"
#include "Layer.h"
#include "Position.h"
#include "types.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <sys/types.h>

class membuf : public std::basic_streambuf<char> {
public:
  membuf(const uint8_t *p, size_t l) {
    setg((char *)p, (char *)p, (char *)p + l);
  }
};

class memstream : public std::istream {
public:
  memstream(const uint8_t *p, size_t l)
      : std::istream(&_buffer), _buffer(p, l) {
    rdbuf(&_buffer);
  }

private:
  membuf _buffer;
};

Value win_eval(TB_RESULT result, Value score, Position pos);
Value tempo_white(Position pos);
Value tempo_black(Position pos);

constexpr static size_t ALIGNMENT = 64;

template <int OutDim> struct alignas(64) Accumulator {
  alignas(64) int16_t black_acc[OutDim] = {0};
  alignas(64) int16_t white_acc[OutDim] = {0};
  int16_t psqt{0};
  int16_t *ft_biases;
  int16_t *ft_weights;

  int size;
  Position previous_black, previous_white;
  std::array<int, 32> removed_features;
  std::array<int, 32> active_features;
  // testing some stuff

  ~Accumulator();

  void update(Color per, Position after);

  void update(Position after);

  void apply(Color color, Position before, Position after);

  void refresh();

  void load_weights(std::istream &stream);

  uint8_t *forward(uint8_t *in, const Position &next);
};

template <int L1, int L2, int L3> struct Network {
  int max_units{0};
  Accumulator<2 * L1> accumulator;
  QLayer<L1, L2, Activation::SqRelu> first;
  QLayer<L2, L3, Activation ::SqRelu> second;
  QLayer<L3, 1> output;
  alignas(64) uint8_t input[L1 + L2 + L3 + 1] = {0};

  void load_bucket(std::string file);

  void load_from_array(const unsigned char *, size_t size);

  int32_t *compute_incre_forward_pass(Position next);

  int evaluate(Position pos, int ply, int shuffle);

  int get_raw_eval(Position pos);

  int operator[](int index);

  friend class Accumulator<2 * L1>;
};

template <int OutDim> void Accumulator<OutDim>::refresh() {
  for (auto i = 0; i < OutDim; ++i) {
    white_acc[i] = ft_biases[i];
    black_acc[i] = ft_biases[i];
  }
  previous_black = Position{};
  previous_white = Position{};
}

template <int OutDim>
void Accumulator<OutDim>::load_weights(std::istream &stream) {
  ft_weights =
      (int16_t *)std_aligned_alloc(ALIGNMENT, (120 * OutDim) * sizeof(int16_t));
  ft_biases = (int16_t *)std_aligned_alloc(ALIGNMENT, OutDim * sizeof(int16_t));
  stream.read((char *)ft_weights, sizeof(int16_t) * (OutDim * 120));
  stream.read((char *)ft_biases, sizeof(int16_t) * (OutDim));

  for (auto i = 0; i < OutDim; ++i) {
    black_acc[i] = ft_biases[i];
    white_acc[i] = ft_biases[i];
  }
}

template <int OutDim> Accumulator<OutDim>::~Accumulator() {
  std_aligned_free(ft_biases);
  std_aligned_free(ft_weights);
}

template <int OutDim>
void Accumulator<OutDim>::apply(Color perp, Position before, Position after) {
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

template <int OutDim>
void Accumulator<OutDim>::update(Color perp, Position after) {
  if (perp == BLACK) {
    apply(perp, previous_black.get_color_flip(), after.get_color_flip());
    previous_black = after;
  } else {
    apply(perp, previous_white, after);
    previous_white = after;
  }
}

template <int OutDim> void Accumulator<OutDim>::update(Position after) {
  update(BLACK, after);
  update(WHITE, after);
}

template <int OutDim>
uint8_t *Accumulator<OutDim>::forward(uint8_t *in, const Position &next) {
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

template <int L1, int L2, int L3>
void Network<L1, L2, L3>::load_bucket(std::string file) {

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
template <int L1, int L2, int L3>
void Network<L1, L2, L3>::load_from_array(const unsigned char *data,
                                          size_t size) {
  memstream stream(data, size);
  accumulator.load_weights(stream);
  first.load_params(stream);
  second.load_params(stream);
  output.load_params(stream);
}

template <int L1, int L2, int L3>
int32_t *Network<L1, L2, L3>::compute_incre_forward_pass(Position next) {
  auto bucket_index = next.bucket_index();
  auto *out = accumulator.forward(input, next);

  out = first.forward(out, bucket_index);
  out = second.forward(out, bucket_index);
  return output.forward(out, bucket_index);
}

template <int L1, int L2, int L3>
int Network<L1, L2, L3>::operator[](int index) {
  return input[index];
}

template <int L1, int L2, int L3>
int Network<L1, L2, L3>::evaluate(Position pos, int ply, int shuffle) {

  auto nnue = *compute_incre_forward_pass(pos);

  return nnue;
}

template <int L1, int L2, int L3>
int Network<L1, L2, L3>::get_raw_eval(Position pos) {

  const auto nnue = *compute_incre_forward_pass(pos);
  auto eval = (nnue);
  return eval;
}

#endif // READING_NETWORK_H
