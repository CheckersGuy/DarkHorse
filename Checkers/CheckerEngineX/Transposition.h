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
  int tt_capture = -1; // TT move is a capture: its index in the generated move list
  Value score{-SCORE_INFINITE};
  Value static_eval{-SCORE_INFINITE};
  uint8_t depth{0};
  uint8_t flag{Flag::None};
  bool ttPv{false};
};

// 12 bytes. The 64-bit key is checked with 40 bits: the high 32 bits (key, stored
// XORed with a checksum of the entry's data) and the low 8 bits (key8). The checksum
// makes entries torn by concurrent writes of two threads (key of one store, data of
// another) fail the check, so they are misses instead of wrong hits.
struct Entry {
  uint32_t key{0u};                     // 4 bytes: high key bits ^ entry_check()
  int16_t value{-SCORE_INFINITE};       // 2 bytes
  int16_t static_eval{-SCORE_INFINITE}; // 2 bytes
  MoveEncoding best_move;               // 1 byte
  uint8_t age : 6 = 0;                  // 1 byte
  uint8_t flag : 2 = 0;
  uint8_t depth : 7; // 1 byte
  uint8_t ttPv : 1;
  uint8_t key8{0u}; // 1 byte: low key bits (formerly padding)
};
static_assert(sizeof(Entry) == 12, "TT entry must stay 12 bytes");

// checksum of everything but the age (probes refresh the age in place)
inline uint32_t entry_check(const Entry &e) {
  const uint32_t a = (uint32_t)(uint16_t)e.value | ((uint32_t)(uint16_t)e.static_eval << 16);
  const uint32_t b = (uint32_t)e.best_move.raw() | ((uint32_t)e.flag << 8) |
                     ((uint32_t)e.depth << 10) | ((uint32_t)e.ttPv << 17) |
                     ((uint32_t)e.key8 << 18);
  const uint32_t h = a * 0x9E3779B1u ^ b * 0x85EBCA77u;
  return h ^ (h >> 15);
}

#ifdef TT_CLUSTER48
// old layout: 4 entries in 48 bytes (every second cluster spans two cache lines)
constexpr size_t bucket_size = 4;
struct Cluster {
  std::array<Entry, bucket_size> ent;
};
static_assert(sizeof(Cluster) == 48);
#else
// one cache line per cluster: 5 entries (60 bytes) + 4 bytes padding
constexpr size_t bucket_size = 5;
struct alignas(64) Cluster {
  std::array<Entry, bucket_size> ent;
  uint8_t pad[4]{};
};
static_assert(sizeof(Cluster) == 64);
#endif

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

  void resize_in_mb(size_t mb);

  // tt_capture >= 0: the best move is a capture, stored as its index in the
  // generated move list (tt_move must then be empty)
  void store_hash(bool in_pv, Value value, Value static_eval, uint64_t key,
                  Flag flag, uint8_t depth, Move tt_move, bool ttPV,
                  int tt_capture = -1);

  bool find_hash(uint64_t key, NodeInfo &info) const;

  void prefetch(uint64_t key);

  // new game without clearing (clearing a large table costs the first move's time):
  // all existing entries become maximally old (half the 6-bit age range), so the
  // replacement overwrites them first; hits on them stay valid (same positions)
  void age_all() { age_counter = (age_counter + 32) & 63; }

  int get_size_in_mb(); // returns size of the table in MB
};

extern Transposition TT;

#endif // CHECKERSTEST_TRANSPOSITION_H
