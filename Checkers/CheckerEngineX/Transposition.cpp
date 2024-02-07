//
// Created by Robin on 13.06.2017.
//

#include "Transposition.h"
#include "types.h"
#include <cstdint>

Transposition TT;

Transposition::Transposition(size_t capacity) {
  size_t size = 1u << capacity;
  resize(size);
}

void Transposition::resize(size_t capa) {
  capacity = 1u << capa;
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

void Transposition::store_hash(bool in_pv, Value value, Value static_eval,
                               uint64_t key, Flag flag, uint8_t depth,
                               Move tt_move) {
  assert(std::abs(value) <= EVAL_INFINITE);
  assert(!tt_move.is_capture());
  const auto index = (key) & (get_capacity() - 1u);
  Cluster &cluster = this->entries[index];
  const uint32_t lock = (key >> 32u);
  auto &replace = cluster.ent[0];
  int best_score = 100000;

  for (auto i = 0; i < bucket_size; ++i) {
    auto &entry = cluster.ent[i];
    if (entry.flag == Flag::None || entry.key == lock) {
      replace = entry;
      break;
    }
    const int age_entry = age_counter - entry.age;
    int score = 5 * entry.depth - 7 * std::max(age_entry, 0);
    if (score < best_score) {
      best_score = score;
      replace = entry;
    }
  }
  // the move we are storing
  MoveEncoding store_move = replace.best_move;
  if (replace.key != lock || !tt_move.is_empty()) {
    store_move = MoveEncoding(tt_move);
  }
  if (flag == TT_EXACT || replace.key != lock ||
      depth + 4 + 2 * in_pv > replace.depth) {
    replace.key = lock;
    replace.best_move = store_move;
    replace.flag = flag;
    replace.depth = depth;
    replace.age = age_counter;
    replace.value = value;
    replace.static_eval = static_eval;
    return;
  }
  if (replace.key == key && replace.best_move.get_move().is_empty()) {
    replace.best_move = store_move;
  }
}

bool Transposition::find_hash(uint64_t key, NodeInfo &info) const {
  const auto index = key & (get_capacity() - 1u);
  const uint32_t currKey = key >> 32u;

  for (auto i = 0; i < bucket_size; ++i) {
    if (this->entries[index].ent[i].key == key) {
      this->entries[index].ent[i].age = age_counter;
    }
  }
  for (int i = 0; i < bucket_size; ++i) {
    if (this->entries[index].ent[i].key == currKey) {
      info.tt_move = this->entries[index].ent[i].best_move.get_move();
      info.depth = this->entries[index].ent[i].depth;
      info.flag = this->entries[index].ent[i].flag;
      info.score = this->entries[index].ent[i].value;
      info.static_eval = this->entries[index].ent[i].static_eval;
      return true;
    }
  }
  return false;
}

void Transposition::prefetch(uint64_t key) {
  auto index = key & (capacity - 1);
#if defined(_MSC_VER)
  _mm_prefetch((char *)&entries[index(key)], _MM_HINT_T0);
#else
  __builtin_prefetch(&entries[index]);
#endif
}
