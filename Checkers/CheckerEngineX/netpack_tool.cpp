// netpack_tool: packs a quantised net (.quant) into the compressed .dhn format of
// NetPack.h and checks the round trip with the engine's decoder.
//
//   NetPack pack   NET.quant L1 L2 L3 OUT NB NET.dhn   (verifies the result)
//   NetPack unpack NET.dhn NET.quant
//   NetPack check  NET.dhn NET.quant [NET2.dhn NET2.quant ...]   (compare + timing)
//
// e.g. value net:  NetPack pack buckets1024_107.quant 2048 32 32 1 128 value.dhn
// Per tensor the encoder tries all option combinations (median base, number of
// contexts, escape threshold, probability precision) and keeps the smallest result.
#include "NetPack.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <numeric>
#include <string>
#include <thread>

using namespace netpack;

static std::vector<uint8_t> read_file(const char *p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) {
    std::fprintf(stderr, "cannot read %s\n", p);
    std::exit(1);
  }
  return {std::istreambuf_iterator<char>(f), {}};
}

static void write_file(const char *p, const std::vector<uint8_t> &d) {
  std::ofstream f(p, std::ios::binary);
  f.write((const char *)d.data(), (std::streamsize)d.size());
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", p);
    std::exit(1);
  }
}

template <typename T> static void put_le(std::vector<uint8_t> &o, T v) {
  for (size_t k = 0; k < sizeof(T); ++k)
    o.push_back((uint8_t)((uint64_t)v >> (8 * k)));
}

static void put_varint(std::vector<uint8_t> &o, uint32_t v) {
  while (v >= 0x80) {
    o.push_back((uint8_t)(v | 0x80));
    v >>= 7;
  }
  o.push_back((uint8_t)v);
}

static void put_nibbles(std::vector<uint8_t> &o, const std::vector<uint8_t> &v) {
  for (size_t i = 0; i < v.size(); i += 2)
    o.push_back((uint8_t)(v[i] | ((i + 1 < v.size() ? v[i + 1] : 0) << 4)));
}

// 16 quantile bins by rank of v (ties by index)
static std::vector<uint8_t> rank_bins(const std::vector<double> &v) {
  std::vector<size_t> idx(v.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(), [&](size_t a, size_t b) { return v[a] < v[b]; });
  std::vector<uint8_t> bins(v.size());
  for (size_t k = 0; k < idx.size(); ++k)
    bins[idx[k]] = (uint8_t)(k * 16 / idx.size());
  return bins;
}

struct BitWriter {
  std::vector<uint8_t> out;
  uint64_t nbits = 0;
  void put(uint64_t v, int n) { // MSB first
    for (int k = n - 1; k >= 0; --k, ++nbits) {
      if ((nbits & 7) == 0)
        out.push_back(0);
      out.back() |= (uint8_t)(((v >> k) & 1) << (7 - (nbits & 7)));
    }
  }
};

// frequencies summing to M (every used symbol >= 1); empty -> symbol 0 gets M
static bool normalize(const std::vector<uint64_t> &cnt, uint32_t M, std::vector<uint32_t> &f) {
  const uint64_t total = std::accumulate(cnt.begin(), cnt.end(), (uint64_t)0);
  f.assign(cnt.size(), 0);
  if (total == 0) {
    f[0] = M;
    return true;
  }
  std::vector<std::pair<double, uint32_t>> rem;
  int64_t sum = 0;
  uint32_t used = 0;
  for (uint32_t s = 0; s < cnt.size(); ++s)
    if (cnt[s]) {
      const double exact = (double)cnt[s] * M / (double)total;
      f[s] = std::max<uint32_t>(1, (uint32_t)exact);
      rem.push_back({exact - f[s], s});
      sum += f[s];
      ++used;
    }
  if (used > M)
    return false;
  std::sort(rem.begin(), rem.end(), [](auto &a, auto &b) { return a.first > b.first; });
  for (size_t i = 0; sum < (int64_t)M; i = (i + 1) % rem.size(), ++sum)
    f[rem[i].second]++;
  while (sum > (int64_t)M) { // take from the largest frequencies
    uint32_t best = 0;
    for (uint32_t s = 0; s < f.size(); ++s)
      if (f[s] > f[best])
        best = s;
    f[best]--;
    --sum;
  }
  return true;
}

// NETPACK_MAXPB: highest table precision tried; 12 bits keeps the decoding tables in
// the cache (15 bits would save ~0.07 % but decodes much slower)
static int g_maxpb = 12;

// NETPACK_RANDOM=seed: random options per tensor instead of the best (round-trip
// tests of the decoder across all option combinations)
static uint64_t g_random = 0;
static uint64_t next_random() {
  g_random ^= g_random << 13;
  g_random ^= g_random >> 7;
  g_random ^= g_random << 17;
  return g_random;
}

struct Options {
  int base, K, lt, pb;
};

// one tensor with fixed options; base_blob = serialised median plane (if base)
static std::vector<uint8_t> encode_with(uint32_t G, uint32_t R, uint32_t C,
                                        const std::vector<int64_t> &vals,
                                        const std::vector<int64_t> &base,
                                        const std::vector<uint8_t> &base_blob, Options op) {
  const size_t n = vals.size();
  std::vector<int64_t> X(n);
  for (size_t i = 0; i < n; ++i)
    X[i] = vals[i] - (op.base ? base[i % ((size_t)R * C)] : 0);
  std::vector<double> rs((size_t)G * R, 0), cs(C, 0);
  for (size_t i = 0; i < n; ++i) {
    const double a = std::fabs((double)X[i]);
    rs[i / C] += a;
    cs[i % C] += a;
  }
  const std::vector<uint8_t> rowbin = rank_bins(rs), colbin = rank_bins(cs);
  // context map: cells (row bin, column bin) sorted by mean |x|, K equal-count groups
  std::vector<double> cell_sum(256, 0);
  std::vector<uint64_t> cell_n(256, 0);
  for (size_t i = 0; i < n; ++i) {
    const int cell = rowbin[i / C] * 16 + colbin[i % C];
    cell_sum[cell] += std::fabs((double)X[i]);
    cell_n[cell]++;
  }
  std::vector<int> cells;
  for (int c = 0; c < 256; ++c)
    if (cell_n[c])
      cells.push_back(c);
  std::stable_sort(cells.begin(), cells.end(), [&](int a, int b) {
    return cell_sum[a] / cell_n[a] < cell_sum[b] / cell_n[b];
  });
  std::vector<uint8_t> map(256, 0);
  {
    uint64_t before = 0;
    for (int c : cells) {
      const double mid = before + cell_n[c] / 2.0;
      map[c] = (uint8_t)std::min<double>(op.K - 1, std::floor(mid * op.K / n));
      before += cell_n[c];
    }
  }
  // symbols
  const uint32_t T2 = 2u << op.lt, A = alphabet(op.lt), M = 1u << op.pb;
  std::vector<uint32_t> sym(n);
  std::vector<uint8_t> ctx(n);
  BitWriter raw;
  std::vector<std::vector<uint64_t>> cnt(op.K, std::vector<uint64_t>(A, 0));
  for (size_t i = 0; i < n; ++i) {
    const int64_t v = X[i];
    const uint64_t mag = (uint64_t)(v < 0 ? -v : v);
    uint32_t s;
    if (mag < (T2 >> 1)) {
      s = v < 0 ? (uint32_t)(2 * mag - 1) : (uint32_t)(2 * mag);
    } else {
      const int cls = ilog2(mag);
      if (cls > MAX_CLS)
        return {};
      s = T2 + (uint32_t)(cls - op.lt);
      raw.put(mag & ((1ull << cls) - 1), cls);
      raw.put(v < 0, 1);
    }
    sym[i] = s;
    ctx[i] = map[rowbin[i / C] * 16 + colbin[i % C]];
    cnt[ctx[i]][s]++;
  }
  std::vector<std::vector<uint32_t>> f(op.K), cum(op.K);
  for (int k = 0; k < op.K; ++k) {
    if (!normalize(cnt[k], M, f[k]))
      return {};
    cum[k].assign(A, 0);
    for (uint32_t s = 1; s < A; ++s)
      cum[k][s] = cum[k][s - 1] + f[k][s - 1];
  }
  // rANS with two interleaved states (value i: state i % 2), encoded backwards into
  // a reversed buffer; 16-bit renormalisation
  std::vector<uint8_t> rb;
  uint64_t xs[2] = {RANS_L, RANS_L};
  for (size_t i = n; i-- > 0;) {
    uint64_t &x = xs[i & 1];
    const uint64_t fr = f[ctx[i]][sym[i]], st = cum[ctx[i]][sym[i]];
    const uint64_t x_max = ((RANS_L >> op.pb) << 32) * fr;
    if (x >= x_max) { // reversed buffer: little-endian 32-bit word in the final stream
      for (int b = 3; b >= 0; --b)
        rb.push_back((uint8_t)(x >> (8 * b)));
      x >>= 32;
    }
    x = ((x / fr) << op.pb) + (x % fr) + st;
  }
  for (int k = 1; k >= 0; --k) // final order: state 0, state 1 (little endian)
    for (int b = 7; b >= 0; --b)
      rb.push_back((uint8_t)(xs[k] >> (8 * b)));
  std::reverse(rb.begin(), rb.end());
  // serialise
  std::vector<uint8_t> o;
  put_le<uint8_t>(o, (uint8_t)op.base);
  put_le<uint8_t>(o, (uint8_t)op.K);
  put_le<uint8_t>(o, (uint8_t)op.lt);
  put_le<uint8_t>(o, (uint8_t)op.pb);
  put_le<uint32_t>(o, G);
  put_le<uint32_t>(o, R);
  put_le<uint32_t>(o, C);
  if (op.K > 1) {
    put_nibbles(o, rowbin);
    put_nibbles(o, colbin);
    o.insert(o.end(), map.begin(), map.end());
  }
  if (op.base)
    o.insert(o.end(), base_blob.begin(), base_blob.end());
  for (int k = 0; k < op.K; ++k) {
    uint32_t used = 0;
    for (uint32_t s = 0; s < A; ++s)
      used += f[k][s] > 0;
    if (used > 65535)
      return {};
    put_le<uint16_t>(o, (uint16_t)used);
    int64_t prev = -1;
    for (uint32_t s = 0; s < A; ++s)
      if (f[k][s]) {
        put_varint(o, (uint32_t)(prev < 0 ? s : s - prev - 1));
        put_varint(o, f[k][s]);
        prev = s;
      }
  }
  put_le<uint64_t>(o, raw.out.size());
  o.insert(o.end(), raw.out.begin(), raw.out.end());
  put_le<uint64_t>(o, rb.size());
  o.insert(o.end(), rb.begin(), rb.end());
  return o;
}

// tries the option grid in parallel, returns the smallest serialisation
static std::vector<uint8_t> encode_best(uint32_t G, uint32_t R, uint32_t C,
                                        const std::vector<int64_t> &vals, const char *name,
                                        bool verbose) {
  std::vector<int64_t> base;
  std::vector<uint8_t> base_blob;
  if (G >= 2) { // median over the group axis (lower middle element), coded once
    base.resize((size_t)R * C);
    std::vector<int64_t> tmp(G);
    for (size_t rc = 0; rc < (size_t)R * C; ++rc) {
      for (uint32_t g = 0; g < G; ++g)
        tmp[g] = vals[(size_t)g * R * C + rc];
      std::nth_element(tmp.begin(), tmp.begin() + G / 2, tmp.end());
      base[rc] = tmp[G / 2];
    }
    base_blob = encode_best(1, R, C, base, "base", false);
  }
  std::vector<Options> grid;
  for (int b = 0; b <= (G >= 2 ? 1 : 0); ++b)
    for (int K : {1, 2, 4, 8, 16, 24, 32, 48, 64})
      for (int lt = 1; lt <= 12; ++lt)
        for (int pb = 10; pb <= g_maxpb; ++pb)
          if ((uint64_t)K * 16 <= std::max<uint64_t>(16, vals.size() / 64) || K == 1)
            grid.push_back({b, K, lt, pb});
  if (g_random) {
    for (int attempt = 0; attempt < 100; ++attempt) {
      const Options op = grid[next_random() % grid.size()];
      std::vector<uint8_t> o = encode_with(G, R, C, vals, base, base_blob, op);
      if (!o.empty())
        return o;
    }
    std::fprintf(stderr, "no valid random encoding for %s\n", name);
    std::exit(2);
  }
  std::vector<size_t> size(grid.size(), SIZE_MAX);
  std::atomic<size_t> next{0};
  std::vector<std::thread> th;
  const unsigned nt = std::max(1u, std::thread::hardware_concurrency());
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&] {
      for (size_t i; (i = next++) < grid.size();) {
        const auto o = encode_with(G, R, C, vals, base, base_blob, grid[i]);
        if (!o.empty())
          size[i] = o.size();
      }
    });
  for (auto &t : th)
    t.join();
  const size_t best = (size_t)(std::min_element(size.begin(), size.end()) - size.begin());
  if (size[best] == SIZE_MAX) {
    std::fprintf(stderr, "no valid encoding for %s\n", name);
    std::exit(2);
  }
  const Options op = grid[best];
  std::vector<uint8_t> o = encode_with(G, R, C, vals, base, base_blob, op);
  if (verbose)
    std::printf("  %-12s %9zu values -> %9zu bytes (%.3f bits/value; base %d, %d contexts, "
                "T %d, %d-bit tables)\n",
                name, vals.size(), o.size(), 8.0 * o.size() / vals.size(), op.base, op.K,
                1 << op.lt, op.pb);
  return o;
}

// file order: acc weights, acc biases, then per layer and bucket weights + biases
static std::vector<uint8_t> assemble(const Arch &a, const std::vector<int64_t> *t) {
  uint32_t shp[8][4];
  tensor_shapes(a, shp);
  size_t total = 0;
  for (int k = 0; k < 8; ++k)
    total += t[k].size() * shp[k][3];
  std::vector<uint8_t> out(total);
  uint8_t *o = out.data();
  auto put = [&](const int64_t *v, size_t n, int bytes) {
    for (size_t i = 0; i < n; ++i) {
      const uint64_t u = (uint64_t)v[i]; // two's complement, little endian
      for (int b = 0; b < bytes; ++b)
        *o++ = (uint8_t)(u >> (8 * b));
    }
  };
  for (int k = 0; k < 2; ++k)
    put(t[k].data(), t[k].size(), (int)shp[k][3]);
  for (int l = 2; l < 8; l += 2) {
    const size_t nw = (size_t)shp[l][1] * shp[l][2], nb = shp[l + 1][2];
    for (uint32_t b = 0; b < a.NB; ++b) {
      put(t[l].data() + b * nw, nw, (int)shp[l][3]);
      put(t[l + 1].data() + b * nb, nb, (int)shp[l + 1][3]);
    }
  }
  return out;
}

// splits a .quant file into the 8 tensors of Network<L1,L2,L3,Out,NB>
static void split(const std::vector<uint8_t> &d, const Arch &a, std::vector<int64_t> *t) {
  uint32_t shp[8][4];
  tensor_shapes(a, shp);
  size_t o = 0;
  auto get = [&](int bytes) -> int64_t {
    if (o + bytes > d.size()) {
      std::fprintf(stderr, "file too short for this architecture\n");
      std::exit(1);
    }
    uint64_t u = 0;
    for (int b = 0; b < bytes; ++b)
      u |= (uint64_t)d[o + b] << (8 * b);
    o += bytes;
    const int sh = 64 - 8 * bytes; // sign extension
    return (int64_t)(u << sh) >> sh;
  };
  for (int k = 0; k < 2; ++k) {
    t[k].resize((size_t)shp[k][0] * shp[k][1] * shp[k][2]);
    for (auto &v : t[k])
      v = get((int)shp[k][3]);
  }
  for (int l = 2; l < 8; l += 2) {
    const size_t nw = (size_t)shp[l][1] * shp[l][2], nb = shp[l + 1][2];
    t[l].resize(a.NB * nw);
    t[l + 1].resize(a.NB * nb);
    for (uint32_t b = 0; b < a.NB; ++b) {
      for (size_t i = 0; i < nw; ++i)
        t[l][b * nw + i] = get((int)shp[l][3]);
      for (size_t i = 0; i < nb; ++i)
        t[l + 1][b * nb + i] = get((int)shp[l + 1][3]);
    }
  }
  if (o != d.size()) {
    std::fprintf(stderr, "file size %zu does not match the architecture (%zu bytes)\n", d.size(), o);
    std::exit(1);
  }
}

static Arch arch_args(char **argv) {
  return Arch{(uint32_t)std::atoi(argv[0]), (uint32_t)std::atoi(argv[1]),
              (uint32_t)std::atoi(argv[2]), (uint32_t)std::atoi(argv[3]),
              (uint32_t)std::atoi(argv[4])};
}

static const char *names[8] = {"acc weights", "acc biases", "l1 weights", "l1 biases",
                               "l2 weights",  "l2 biases",  "out weights", "out biases"};

int main(int argc, char **argv) {
  const std::string cmd = argc > 1 ? argv[1] : "";
  if (const char *e = std::getenv("NETPACK_RANDOM"))
    g_random = std::strtoull(e, nullptr, 10) * 0x9E3779B97F4A7C15ull | 1;
  if (const char *e = std::getenv("NETPACK_MAXPB"))
    g_maxpb = std::max(10, std::min(15, std::atoi(e)));
  if (cmd == "pack" && argc == 9) {
    const std::vector<uint8_t> d = read_file(argv[2]);
    const Arch a = arch_args(argv + 3);
    std::vector<int64_t> t[8];
    split(d, a, t);
    if (assemble(a, t) != d) {
      std::fprintf(stderr, "internal error: split/assemble is not the identity\n");
      return 2;
    }
    uint32_t shp[8][4];
    tensor_shapes(a, shp);
    std::vector<uint8_t> o = {'D', 'H', 'N', '4'};
    put_le<uint64_t>(o, d.size());
    put_le<uint64_t>(o, fnv64(d.data(), d.size()));
    for (uint32_t v : {a.L1, a.L2, a.L3, a.Out, a.NB})
      put_le<uint32_t>(o, v);
    std::printf("%s: Network<%u,%u,%u,%u,%u>\n", argv[2], a.L1, a.L2, a.L3, a.Out, a.NB);
    for (int k = 0; k < 8; ++k) {
      const std::vector<uint8_t> b = encode_best(shp[k][0], shp[k][1], shp[k][2], t[k], names[k], true);
      o.insert(o.end(), b.begin(), b.end());
    }
    const auto t0 = std::chrono::steady_clock::now();
    const std::vector<uint8_t> back = unpack(o.data(), o.size());
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (back != d) {
      std::fprintf(stderr, "ROUND TRIP FAILED: decoded bytes differ from the original\n");
      return 2;
    }
    write_file(argv[8], o);
    std::printf("%s: %zu -> %zu bytes (%.1f %%), round trip identical, decode %.1f ms\n", argv[8],
                d.size(), o.size(), 100.0 * o.size() / d.size(), ms);
    return 0;
  }
  if (cmd == "est" && argc == 9) { // one tensor only (experiments)
    const std::vector<uint8_t> d = read_file(argv[2]);
    const Arch a = arch_args(argv + 3);
    std::vector<int64_t> t[8];
    split(d, a, t);
    uint32_t shp[8][4];
    tensor_shapes(a, shp);
    const int k = std::atoi(argv[8]);
    encode_best(shp[k][0], shp[k][1], shp[k][2], t[k], names[k], true);
    return 0;
  }
  if (cmd == "unpack" && argc == 4) {
    const std::vector<uint8_t> blob = read_file(argv[2]);
    write_file(argv[3], unpack(blob.data(), blob.size()));
    return 0;
  }
  if (cmd == "check" && argc >= 4 && argc % 2 == 0) { // pairs: NET.dhn NET.quant ...
    double total = 0;
    for (int i = 2; i < argc; i += 2) {
      const std::vector<uint8_t> blob = read_file(argv[i]), orig = read_file(argv[i + 1]);
      const auto t0 = std::chrono::steady_clock::now();
      const std::vector<uint8_t> out = unpack(blob.data(), blob.size());
      const double ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      total += ms;
      if (out != orig) {
        std::fprintf(stderr, "ROUND TRIP FAILED: %s differs from %s\n", argv[i], argv[i + 1]);
        return 2;
      }
      std::printf("%s: identical to %s, decode %.1f ms\n", argv[i], argv[i + 1], ms);
    }
    std::printf("total decode time %.1f ms (single thread)\n", total);
    return 0;
  }
  std::fprintf(stderr, "usage: NetPack pack NET.quant L1 L2 L3 OUT NB NET.dhn\n"
                       "       NetPack unpack NET.dhn NET.quant\n"
                       "       NetPack check NET.dhn NET.quant [NET2.dhn NET2.quant ...]\n");
  return 1;
}
