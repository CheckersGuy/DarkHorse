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

// Feature-transformer weights: loaded once, shared read-only by all threads.
template <int OutDim> struct Accumulator {
  constexpr static int out_dimension = OutDim;
  static inline uint32_t num_trials = 0; // used to compute the average
  static inline uint32_t num_nnz =
      0; // used to keep track of the number of nonzero activatios
  static inline uint32_t num_nnz_blocks =
      0; // counts the number of nonzero blocks

  int16_t *ft_biases{nullptr};
  int16_t *ft_weights{nullptr};

  Accumulator() = default;
  Accumulator(const Accumulator &) = delete; // single owner of ft_* buffers
  Accumulator &operator=(const Accumulator &) = delete;
  ~Accumulator();

  void load_weights(std::istream &stream, std::vector<size_t> permutation);

  std::pair<float, float> get_activation_stats();
};

// Incrementally updated accumulator values: one per search thread.
template <int OutDim> struct alignas(64) AccumulatorState {
  alignas(64) int16_t black_acc[OutDim];
  alignas(64) int16_t white_acc[OutDim];
  Position previous_black_flipped, previous_white;
  std::array<int, 32> removed_features;
  std::array<int, 32> active_features;

  void update(const Accumulator<OutDim> &w, Color per, Position after);

  void update(const Accumulator<OutDim> &w, Position after);

  void apply(const Accumulator<OutDim> &w, Color color, Position before,
             Position after);

  void refresh(const Accumulator<OutDim> &w);

  uint8_t *forward(const Accumulator<OutDim> &w, uint8_t *in,
                   const Position &next);
};

template <int L1, int L2, int L3, int Out, int NUM_BUCKETS> struct Network {

  int max_units{0};
  Network() {
    for (auto i = 0; i < 2 * L1; ++i) {
      permutation.emplace_back(i);
    }
  }

  using FirstLayer = SparseLayer<L1, L2, NUM_BUCKETS>;
  using SecondLayer = QLayer<L2, L3, Activation::SqRelu, NUM_BUCKETS>;
  using OutLayer = QLayer<L3, Out, Activation::Id, NUM_BUCKETS>;

  // ---- weights: shared read-only by all threads ----
  Accumulator<2 * L1> accumulator;
  FirstLayer first_layer;
  SecondLayer second_layer;
  OutLayer out_layer;

  // permutations are needed to improve sparsity
  // permutation size should be 2*L1
  std::vector<size_t> permutation;

  // ---- per-thread evaluation state (accumulators + layer scratch buffers) ----
  struct State {
    AccumulatorState<2 * L1> acc;
    alignas(64) uint8_t input[(L1 + L2 + L3 + Out) + 128] = {0};
    alignas(64) int32_t first_buffer[FirstLayer::PadOutDim] = {0};
    alignas(64) int32_t second_buffer[SecondLayer::PadOutDim] = {0};
    alignas(64) int32_t out_buffer[OutLayer::PadOutDim] = {0};
    alignas(64) uint16_t nnz[FirstLayer::NNZ_SIZE] = {0};
    bool ready{false}; // accumulators initialised from the biases
  };

  void load_permutation(std::string file);
  void load_permutation_from_array(const unsigned char *, size_t size);

  void load_bucket(std::string file);

  void load_from_array(const unsigned char *, size_t size);

  void refresh(State &st) const;

  int32_t *compute_incre_forward_pass(State &st, Position next) const;

  int evaluate(State &st, Position pos, int ply, int shuffle) const;

  int32_t *get_raw_eval(State &st, Position pos) const;

  // Convenience for single-threaded callers (tools, book code): a lazily
  // initialised state per OS thread. The search uses its own per-thread State.
  State &default_state() const {
    thread_local State st;
    if (!st.ready)
      refresh(st);
    return st;
  }
  void refresh() const { refresh(default_state()); }
  int evaluate(Position pos, int ply, int shuffle) const {
    return evaluate(default_state(), pos, ply, shuffle);
  }
  int32_t *get_raw_eval(Position pos) const {
    return get_raw_eval(default_state(), pos);
  }
};

template <int OutDim>
void AccumulatorState<OutDim>::refresh(const Accumulator<OutDim> &w) {
  for (auto i = 0; i < OutDim; ++i) {
    white_acc[i] = w.ft_biases[i];
    black_acc[i] = w.ft_biases[i];
  }
  previous_black_flipped = Position{};
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
}

template <int OutDim> Accumulator<OutDim>::~Accumulator() {
  std_aligned_free(ft_biases);
  std_aligned_free(ft_weights);
}

template <int OutDim>
void AccumulatorState<OutDim>::apply(const Accumulator<OutDim> &w, Color perp,
                                     Position before, Position after) {
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
      const __m256i *weights = reinterpret_cast<const __m256i *>(
          w.ft_weights + OutDim * active_features[i]);

      for (auto j = 0; j < num_regs; ++j) {
        regs[j] = _mm256_add_epi16(
            _mm256_load_si256(weights + j + k * num_regs), regs[j]);
      }
    }

    for (auto i = 0; i < num_removed; ++i) {
      const __m256i *weights = reinterpret_cast<const __m256i *>(
          w.ft_weights + OutDim * removed_features[i]);
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
void AccumulatorState<OutDim>::update(const Accumulator<OutDim> &w, Color perp,
                                      Position after) {
  if (perp == BLACK) {
    const Position after_flipped = after.get_color_flip();
    apply(w, perp, previous_black_flipped, after.get_color_flip());
    previous_black_flipped = after_flipped;
  } else {
    apply(w, perp, previous_white, after);
    previous_white = after;
  }
}

template <int OutDim>
void AccumulatorState<OutDim>::update(const Accumulator<OutDim> &w,
                                      Position after) {
  update(w, BLACK, after);
  update(w, WHITE, after);
}

template <int OutDim>
uint8_t *AccumulatorState<OutDim>::forward(const Accumulator<OutDim> &w,
                                           uint8_t *in, const Position &next) {
  int16_t *z_previous;
  if (next.color == BLACK) {
    z_previous = black_acc;
  } else {
    z_previous = white_acc;
  }
  update(w, next.color, next);
  Simd::accum_activation8<OutDim>(z_previous, in);
#ifdef SPARSEOPT
  for (auto i = 0; i < OutDim / 2; ++i) {
    Accumulator<OutDim>::num_nnz += (in[i] != 0);
  }
  auto *blocks = reinterpret_cast<uint32_t *>(in);
  for (auto i = 0; i < OutDim / 8; ++i) {
    Accumulator<OutDim>::num_nnz_blocks += (blocks[i] != 0);
  }
  Accumulator<OutDim>::num_trials += 1;
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

template <int L1, int L2, int L3, int Out, int NUM_BUCKETS>
void Network<L1, L2, L3, Out, NUM_BUCKETS>::load_permutation(std::string file) {
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

template <int L1, int L2, int L3, int Out, int NUM_BUCKETS>
void Network<L1, L2, L3, Out, NUM_BUCKETS>::load_permutation_from_array(
    const unsigned char *data, size_t size) {
  memstream stream(data, size);
  const auto file_size = size;
  const auto num_items = file_size / sizeof(size_t);
  std::vector<size_t> result(num_items);
  stream.read((char *)&result[0], file_size);

  permutation = result;
}

template <int L1, int L2, int L3, int Out, int NUM_BUCKETS>
void Network<L1, L2, L3, Out, NUM_BUCKETS>::load_bucket(std::string file) {

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
template <int L1, int L2, int L3, int Out, int NUM_BUCKETS>
void Network<L1, L2, L3, Out, NUM_BUCKETS>::load_from_array(
    const unsigned char *data, size_t size) {
  memstream stream(data, size);
  accumulator.load_weights(stream, permutation);
  first_layer.load_params(stream, permutation);
  second_layer.load_params(stream);
  out_layer.load_params(stream);
  // A network file that does not match the template parameters would leave
  // weights uninitialised (failed reads) or silently ignore trailing data.
  if (stream.fail() || stream.peek() != std::char_traits<char>::eof()) {
    std::cerr << "Network<" << L1 << ", " << L2 << ", " << L3 << ", " << Out
              << ", " << NUM_BUCKETS << ">: embedded network data of " << size
              << " bytes does not match the architecture" << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

template <int L1, int L2, int L3, int Out, int NUM_BUCKETS>
void Network<L1, L2, L3, Out, NUM_BUCKETS>::refresh(State &st) const {
  st.acc.refresh(accumulator);
  st.ready = true;
}

template <int L1, int L2, int L3, int Out, int NUM_BUCKETS>
int32_t *Network<L1, L2, L3, Out, NUM_BUCKETS>::compute_incre_forward_pass(
    State &st, Position next) const {

  auto bucket_index = next.bucket_index<NUM_BUCKETS>();
  uint8_t *out = st.acc.forward(accumulator, st.input, next);

  int32_t *output = nullptr;

  out = first_layer.forward(out, bucket_index, st.first_buffer, st.nnz);
  out = second_layer.forward(out, bucket_index, st.second_buffer);
  output = out_layer.forward(out, bucket_index, st.out_buffer);
  return output;
}

template <int L1, int L2, int L3, int Out, int NUM_BUCKETS>
int Network<L1, L2, L3, Out, NUM_BUCKETS>::evaluate(State &st, Position pos,
                                                    int ply,
                                                    int shuffle) const {

  auto nnue = *compute_incre_forward_pass(st, pos);

  return nnue;
}

template <int L1, int L2, int L3, int Out, int NUM_BUCKETS>
int32_t *
Network<L1, L2, L3, Out, NUM_BUCKETS>::get_raw_eval(State &st,
                                                    Position pos) const {

  return compute_incre_forward_pass(st, pos);
}

#endif // READING_NETWORK_H
