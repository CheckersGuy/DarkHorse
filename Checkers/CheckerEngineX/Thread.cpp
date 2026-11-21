#include "Thread.h"
#include <algorithm>

std::vector<std::unique_ptr<ThreadData>> threads = [] {
  std::vector<std::unique_ptr<ThreadData>> v;
  v.emplace_back(std::make_unique<ThreadData>());
  return v;
}();

void set_num_threads(int n) {
  n = std::max(1, n);
  while ((int)threads.size() > n)
    threads.pop_back();
  while ((int)threads.size() < n) {
    threads.emplace_back(std::make_unique<ThreadData>());
    threads.back()->id = (int)threads.size() - 1;
  }
}

int get_num_threads() { return (int)threads.size(); }

ThreadData &main_td() { return *threads[0]; }

uint64_t total_nodes() {
  uint64_t sum = 0;
  for (const auto &t : threads)
    sum += t->nodes.load(std::memory_order_relaxed);
  return sum;
}
