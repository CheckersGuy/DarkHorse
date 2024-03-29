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
  Value static_eval{0};
  uint8_t depth{0};
  uint8_t flag{Flag::None};
};

struct Entry {
  uint32_t key{0u};       // 4 bytes
  int16_t value{0};       // 2 bytes
  int16_t static_eval;    // 2 bytes
  MoveEncoding best_move; // 1byte
  uint8_t age : 6 = 0;    // 1 bytes
  uint8_t flag : 2 = 0;
  uint8_t depth{0}; // 1 byte
                    //
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

  void store_hash(bool in_pv, Value value, Value static_eval, uint64_t key,
                  Flag flag, uint8_t depth, Move tt_move);

  bool find_hash(uint64_t key, NodeInfo &info) const;

  void prefetch(uint64_t key);

  int get_size_in_mb(); // returns size of the table in MB
};

extern Transposition TT;

#endif // CHECKERSTEST_TRANSPOSITION_H
