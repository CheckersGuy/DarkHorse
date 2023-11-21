//
// Created by Robin on 13.06.2017.
//

#ifndef CHECKERSTEST_TRANSPOSITION_H
#define CHECKERSTEST_TRANSPOSITION_H

#include "Move.h"
#include "MoveListe.h"
#include "types.h"
#include <cstdint>
#include <iostream>
#include <string>

struct NodeInfo {
  Move tt_move;
  Value score{0};
  uint8_t depth{0};
  uint8_t flag{Flag::None};
};

struct MoveEncoding {
  // How to encode empty moves ?
  MoveEncoding(Move move);
  MoveEncoding() = default;
  uint8_t from_index : 6;
  uint8_t direction : 2;
  Move get_move();
  void encode_move(Move move);
};

struct Entry {
  uint32_t key{0u}; // 4 bytes
  int16_t value{0}; // 2 bytes
  MoveEncoding best_move;
  uint8_t age : 6 = 0;  // 1 bytes
  uint8_t flag : 2 = 0; // 1 byte
  uint8_t depth{0};     // 1 byte
};
constexpr size_t bucket_size = 4;

struct Cluster {
  std::array<Entry, bucket_size> ent;
};

class Transposition {

public:
  size_t hashHit{0u};
  uint8_t age_counter{0};
  size_t capacity{0};
  std::unique_ptr<Cluster[]> entries;

public:
  explicit Transposition(size_t length);

  Transposition() = default;

  size_t get_capacity() const;

  uint32_t get_hash_hits() const;

  void clear();

  void resize(size_t capa);

  void store_hash(Value value, uint64_t key, Flag flag, uint8_t depth,
                  Move tt_move);

  bool find_hash(uint64_t key, NodeInfo &info) const;

  void prefetch(uint64_t key);
};

extern Transposition TT;

#endif // CHECKERSTEST_TRANSPOSITION_H
