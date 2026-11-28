//
// Created by Robin on 13.06.2017.
//

#include "Transposition.h"
#include "types.h"
#include <cstdint>

Transposition TT;

Transposition::Transposition(size_t capacity) { resize(capacity); }

void Transposition::resize(size_t capa) {
  capacity = capa;
  entries = std::make_unique<Cluster[]>(capacity);
  clear();
}

size_t Transposition::get_capacity() const { return capacity; }

uint32_t Transposition::get_hash_hits() const { return hashHit; }

void Transposition::clear() {
  hashHit = 0;
  age_counter = 0;
  std::fill(entries.get(), entries.get() + capacity, Cluster{});
}

// does the entry hold this key (40-bit check, torn entries fail)?
static inline bool holds(const Entry &e, uint32_t lock, uint8_t lock8) {
  return e.flag != Flag::None && e.key8 == lock8 && (e.key ^ entry_check(e)) == lock;
}

void Transposition::store_hash(bool in_pv, Value value, Value static_eval,
                               uint64_t key, Flag flag, uint8_t depth,
                               Move tt_move, bool ttPV, int tt_capture) {
  key ^= TT_KEY_SALT; // separate entries for MLH-off searches (types.h)

  assert(!tt_move.is_capture());

  const auto index =
      reduce(static_cast<uint32_t>(key), static_cast<uint32_t>(get_capacity()));
  Cluster &cluster = this->entries[index];
  const uint32_t lock = (key >> 32u);
  const uint8_t lock8 = static_cast<uint8_t>(key);
  auto *replace = &cluster.ent[0];
  bool same_key = false;
  int best_score = 100000;

  for (auto i = 0; i < bucket_size; ++i) {
    auto &entry = cluster.ent[i];
    if (entry.flag == Flag::None) {
      replace = &entry;
      break;
    }
    if (holds(entry, lock, lock8)) {
      replace = &entry;
      same_key = true;
      break;
    }
    const int age_entry = (age_counter - entry.age) & 63;
    int score = TT_DEPTH_WEIGHT * entry.depth - TT_AGE_WEIGHT * age_entry;
    if (score < best_score) {
      best_score = score;
      replace = &entry;
    }
  }
  // the move we are storing
  MoveEncoding store_move = replace->best_move;
  if (!same_key || !tt_move.is_empty() || tt_capture >= 0) {
    store_move = tt_capture >= 0 ? MoveEncoding::capture(tt_capture) : MoveEncoding(tt_move);
  }
  if (flag == TT_EXACT || !same_key ||
      depth + TT_OVERWRITE_MARGIN + 2 * in_pv > replace->depth) {
    replace->best_move = store_move;
    replace->flag = flag;
    replace->depth = static_cast<uint8_t>(std::min(depth, uint8_t{127}));
    replace->age = age_counter;
    replace->value = value;
    replace->static_eval = static_eval;
    replace->ttPv = ttPV;
    replace->key8 = lock8;
    replace->key = lock ^ entry_check(*replace);
    return;
  }
  if (replace->best_move.is_none()) {
    replace->best_move = store_move;
    replace->static_eval = static_eval;
    replace->ttPv = ttPV;
    replace->key = lock ^ entry_check(*replace);
  }
}

bool Transposition::find_hash(uint64_t key, NodeInfo &info) const {
  key ^= TT_KEY_SALT;
  const auto index =
      reduce(static_cast<uint32_t>(key), static_cast<uint32_t>(get_capacity()));
  const uint32_t lock = key >> 32u;
  const uint8_t lock8 = static_cast<uint8_t>(key);
  Cluster &cluster = this->entries[index];
  for (int i = 0; i < bucket_size; ++i) {
    Entry &e = cluster.ent[i];
    if (holds(e, lock, lock8)) {
      if (e.age != age_counter) // no store (cache line traffic) when unchanged
        e.age = age_counter;
      info.tt_move = e.best_move.get_move();
      info.tt_capture = e.best_move.capture_index();
      info.depth = e.depth;
      info.flag = e.flag;
      info.score = e.value;
      info.static_eval = e.static_eval;
      info.ttPv = e.ttPv;
      return true;
    }
  }
  return false;
}

void Transposition::prefetch(uint64_t key) {
  key ^= TT_KEY_SALT;
  const auto index =
      reduce(static_cast<uint32_t>(key), static_cast<uint32_t>(get_capacity()));
  __builtin_prefetch(&entries[index]);
}

int Transposition::get_size_in_mb() {

  const auto bytes = sizeof(Cluster) * capacity;

  return bytes / (1024 * 1024);
}

void Transposition::resize_in_mb(size_t mb) {

  const auto num_entries = mb * (1024 * 1024) / sizeof(Cluster);
  resize(num_entries);
}
