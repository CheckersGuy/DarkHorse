//
// Created by root on 18.04.21.
//

#ifndef READING_NETWORK_H
#define READING_NETWORK_H
#include "Bits.h"
#include "Layer.h"
#include "LinearSparse.h"
#include "Position.h"
#include "types.h"
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <stdlib.h>
#include <sys/types.h>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

template <typename Check, typename... Types> constexpr auto get_unique() {

  if constexpr ((std::is_same_v<Check, Types> || ...)) {
    return get_unique<Types...>();
  } else {

    if constexpr (sizeof...(Types) > 0) {
      return std::tuple_cat(std::tuple<Check>{}, get_unique<Types...>());
    } else {
      return std::tuple<Check>{};
    }
  }
}

template <typename... Types>
constexpr auto get_unique_tuple(std::tuple<Types...>) {
  return get_unique<Types...>();
}

// some more code goes here

template <int counter, int first, int second, int... Args>
constexpr auto get_window_tuple() {

  if constexpr (sizeof...(Args) == 0) {
    return std::tuple<QLayer<first, second>>{};
  } else {

    if constexpr (counter == 0) {
      return std::tuple_cat(std::make_tuple(SparseLayer<first, second>{}),
                            get_window_tuple<counter + 1, second, Args...>());
    } else {
      return std::tuple_cat(
          std::make_tuple(QLayer<first, second, Activation::SqRelu>{}),
          get_window_tuple<counter + 1, second, Args...>());
    }
  }
}

template <typename... Args> constexpr auto getVariant(std::tuple<Args...>) {
  return std::variant<Args...>{};
};

template <int... layers>
using VariantLayerType =
    decltype(getVariant(get_unique_tuple(get_window_tuple<0, layers...>())));

template <class... Ts> struct overloaded : Ts... {
  using Ts::operator()...;
};

template <class... Ts> overloaded(Ts...) -> overloaded<Ts...>;

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
  constexpr static int out_dimension = OutDim;
  static inline uint32_t num_trials = 0; // used to compute the average
  static inline uint32_t num_nnz =
      0; // used to keep track of the number of nonzero activatios
  static inline uint32_t num_nnz_blocks =
      0; // counts the number of nonzero blocks

  alignas(64) int16_t black_acc[OutDim];
  alignas(64) int16_t white_acc[OutDim];
  // used to improve block sparsity
  int16_t *ft_biases;
  int16_t *ft_weights;

  int size;
  Position previous_black, previous_white;
  std::array<int, 32> removed_features;
  std::array<int, 32> active_features;

  ~Accumulator();

  void update(Color per, Position after);

  void update(Position after);

  void apply(Color color, Position before, Position after);

  void refresh();

  void load_weights(std::istream &stream, std::vector<size_t> permutation);

  uint8_t *forward(uint8_t *in, const Position &next);
  std::pair<float, float> get_activation_stats();
};

template <int L1, int L2, int L3, int Out> struct Network {

  int max_units{0};
  Network() {
    for (auto i = 0; i < 2 * L1; ++i) {
      permutation.emplace_back(i);
    }
  }

  Accumulator<2 * L1> accumulator;
  SparseLayer<L1, L2> first_layer;
  QLayer<L2, L3, Activation::SqRelu> second_layer;
  QLayer<L3, Out, Activation::Id> out_layer;

  // permutations are needed to improve sparsity
  // permutation size should be 2*L1
  std::vector<size_t> permutation;

  alignas(64) uint8_t input[(L1 + L2 + L3 + Out) + 128] = {0};

  void load_permutation(std::string file);
  void load_permutation_from_array(const unsigned char *, size_t size);

  void load_bucket(std::string file);

  void load_from_array(const unsigned char *, size_t size);

  int32_t *compute_incre_forward_pass(Position next);

  int evaluate(Position pos, int ply, int shuffle);

  int32_t *get_raw_eval(Position pos);

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
void Accumulator<OutDim>::load_weights(std::istream &stream,
                                       std::vector<size_t> permutation) {
  // TODO need proper error handling here and in other places as well

  if (permutation.size() != OutDim) {
    std::cout << "Permutation Size is not matching OutDim" << std::endl;
    return;
  }

  ft_weights =
      (int16_t *)std_aligned_alloc(ALIGNMENT, (120 * OutDim) * sizeof(int16_t));
  ft_biases = (int16_t *)std_aligned_alloc(ALIGNMENT, OutDim * sizeof(int16_t));

  int16_t *col_buffer = (int16_t *)malloc((OutDim) * sizeof(int16_t));
  int16_t *buffer_biases = (int16_t *)malloc((OutDim) * sizeof(int16_t));

  // permutating the rows first
  size_t index = 0;
  for (auto col = 0; col < 120; col++) {
    stream.read((char *)col_buffer, sizeof(int16_t) * (OutDim));
    for (auto row = 0; row < OutDim; row++) {
      ft_weights[index] = col_buffer[permutation[row]];
      index++;
    }
  }
  stream.read((char *)buffer_biases, sizeof(int16_t) * (OutDim));

  // shuffling the biases

  for (auto i = 0; i < OutDim; i++) {
    // std::cout<<permutation[i]<<std::endl;
    ft_biases[i] = buffer_biases[permutation[i]];
  }

  free(col_buffer);
  free(buffer_biases);
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
  // assert(before.is_legal());
  // assert(after.is_legal());
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
  Simd::accum_activation8<OutDim>(z_previous, in);
#ifdef SPARSEOPT
  for (auto i = 0; i < OutDim / 2; ++i) {
    num_nnz += (in[i] != 0);
  }
  auto *blocks = reinterpret_cast<uint32_t *>(in);
  for (auto i = 0; i < OutDim / 8; ++i) {
    num_nnz_blocks += (blocks[i] != 0);
  }
  num_trials += 1;
#endif
  return in;
}
// used to gather statistics about the activation
template <int OutDim>
std::pair<float, float> Accumulator<OutDim>::get_activation_stats() {
  // Now we can compute the averag
  // Need to check if OutDim is 2*L1
  const auto L1 = OutDim / 2; // OutDim should always be divisible by 2
  const auto NUM_BLOCKS = L1 / 4;
  const auto f_nnz =
      static_cast<float>(num_nnz) / static_cast<float>(num_trials * L1);
  const auto f_nnz_blocks = static_cast<float>(num_nnz_blocks) /
                            static_cast<float>(num_trials * NUM_BLOCKS);

  return std::make_pair(f_nnz, f_nnz_blocks);
}

template <int L1, int L2, int L3, int Out>
void Network<L1, L2, L3, Out>::load_permutation(std::string file) {
  std::filesystem::path p(file);
  const auto file_size = std::filesystem::file_size(p);
  const auto num_items = file_size / sizeof(size_t);
  std::ifstream stream(file, std::ios::binary);

  if (!stream.good()) {
    std::cout << "Could not set up the stream" << std::endl;
    return;
  }

  std::vector<size_t> result(num_items);
  stream.read((char *)&result[0], file_size);
  permutation = result;
}

template <int L1, int L2, int L3, int Out>
void Network<L1, L2, L3, Out>::load_permutation_from_array(
    const unsigned char *data, size_t size) {
  memstream stream(data, size);
  const auto file_size = size;
  const auto num_items = file_size / sizeof(size_t);
  std::vector<size_t> result(num_items);
  stream.read((char *)&result[0], file_size);

  permutation = result;
}

template <int L1, int L2, int L3, int Out>
void Network<L1, L2, L3, Out>::load_bucket(std::string file) {

  std::ifstream stream(file, std::ios::binary);
  if (!stream.good()) {
    std::cerr << "Could not load network file, path " << file << std::endl;
    std::exit(-1);
  }
  accumulator.load_weights(stream, permutation);
  first_layer.load_params(stream, permutation);
  second_layer.load_params(stream);
  out_layer.load_params(stream);
}
template <int L1, int L2, int L3, int Out>
void Network<L1, L2, L3, Out>::load_from_array(const unsigned char *data,
                                               size_t size) {
  memstream stream(data, size);
  accumulator.load_weights(stream, permutation);
  first_layer.load_params(stream, permutation);
  second_layer.load_params(stream);
  out_layer.load_params(stream);
}

template <int L1, int L2, int L3, int Out>
int32_t *Network<L1, L2, L3, Out>::compute_incre_forward_pass(Position next) {

  auto bucket_index = next.bucket_index();
  uint8_t *out = accumulator.forward(input, next);

  int32_t *output = nullptr;

  out = first_layer.forward(out, bucket_index);
  out = second_layer.forward(out, bucket_index);
  output = out_layer.forward(out, bucket_index);
  return output;
}

template <int L1, int L2, int L3, int Out>
int Network<L1, L2, L3, Out>::operator[](int index) {
  return input[index];
}

template <int L1, int L2, int L3, int Out>
int Network<L1, L2, L3, Out>::evaluate(Position pos, int ply, int shuffle) {

  auto nnue = *compute_incre_forward_pass(pos);

  return nnue;
}

template <int L1, int L2, int L3, int Out>
int32_t *Network<L1, L2, L3, Out>::get_raw_eval(Position pos) {

  return compute_incre_forward_pass(pos);
}

#endif // READING_NETWORK_H
