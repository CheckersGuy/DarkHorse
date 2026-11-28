// NetPack.h - lossless codec for the embedded quantised nets (.quant <-> .dhn).
//
// The engine embeds the compressed .dhn files and decodes them at startup
// (netpack::unpack, single-threaded). Decoding rebuilds the exact bytes of the
// original .quant file and verifies them against the size and checksum stored in
// the header before Network::load_from_array sees them; on any mismatch the engine
// stops. Packing tool: netpack_tool.cpp (target NetPack), which also checks the
// round trip with this decoder.
//
// What it exploits (the file is Network<L1,L2,L3,Out,NB>, a list of integer tensors):
//  - every tensor is coded as integer VALUES (int16 accumulator, int8 layer weights,
//    int32 biases) with static rANS (one table lookup per value);
//  - optionally as residual against the median over the bucket axis (the buckets are
//    shared weights + per-bucket deltas);
//  - the value distribution depends on the scale of its row (output neuron of a
//    bucket / accumulator feature) and of its column (input neuron): every tensor
//    has 16 row-scale bins x 16 column-scale bins (side information) mapped to K
//    contexts with their own frequency table;
//  - values with |v| >= T are coded as an escape class (log2) + raw mantissa bits.
//
// Format (little endian):
//   net:    "DHN4" u64 size u64 checksum u32 L1 L2 L3 Out NB, then 8 tensors:
//           acc weights [120][1][2*L1] int16, acc biases [1][1][2*L1] int16,
//           layer1 weights [NB][L2][L1] int8, biases [NB][1][L2] int32,
//           layer2 weights [NB][L3][L2] int8, biases [NB][1][L3] int32,
//           output weights [NB][Out][L3] int8, biases [NB][1][Out] int32
//   tensor: u8 flags (1 = median base) u8 K u8 log2(T) u8 probability bits
//           u32 G R C, [K > 1: row bins (G*R nibbles), column bins (C nibbles),
//           context map (256 x u8, index row bin * 16 + column bin)], [base tensor G=1 (same
//           format)], K frequency tables (u16 used symbols, per symbol varint gap +
//           varint frequency, sum = 2^bits), u64 raw bytes + raw bits (MSB first),
//           u64 rANS bytes + rANS stream (two interleaved 64-bit states, value i
//           uses state i % 2, initial states first, 32-bit little-endian refills)
//   symbols: zigzag(v) for |v| < T, else 2T + log2|v| - log2(T) followed by
//           log2|v| mantissa bits and a sign bit in the raw stream
#ifndef NETPACK_H
#define NETPACK_H

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

namespace netpack {

// FNV-1a over little-endian 64-bit words, then the remaining bytes
inline uint64_t fnv64(const uint8_t *p, size_t n) {
  uint64_t h = 0xcbf29ce484222325ull;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t w = 0;
    for (int k = 7; k >= 0; --k)
      w = (w << 8) | p[i + k];
    h ^= w;
    h *= 0x100000001b3ull;
  }
  for (; i < n; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

[[noreturn]] inline void fail(const char *what) {
  std::cerr << "embedded network data is corrupt (" << what << ")" << std::endl;
  std::exit(EXIT_FAILURE);
}

constexpr uint64_t RANS_L = 1ull << 31; // lower bound of the normalised rANS state
constexpr int MAX_CLS = 32;           // magnitudes up to 2^33 - 1

inline int ilog2(uint64_t v) { // v >= 1
  int k = 0;
  while (v >>= 1)
    ++k;
  return k;
}

// number of symbols for escape threshold T = 2^lt
inline uint32_t alphabet(int lt) { return (2u << lt) + (uint32_t)(MAX_CLS + 1 - lt); }

struct Reader {
  const uint8_t *p, *end;
  template <typename T> T get() {
    if ((size_t)(end - p) < sizeof(T))
      fail("truncated header");
    T v = 0;
    for (size_t k = 0; k < sizeof(T); ++k)
      v |= (T)((T)p[k] << (8 * k));
    p += sizeof(T);
    return v;
  }
  uint32_t varint() {
    uint32_t v = 0;
    for (int shift = 0; shift < 35; shift += 7) {
      const uint8_t b = get<uint8_t>();
      v |= (uint32_t)(b & 0x7F) << shift;
      if (!(b & 0x80))
        return v;
    }
    fail("varint");
  }
  const uint8_t *take(uint64_t n) {
    if ((uint64_t)(end - p) < n)
      fail("truncated data");
    const uint8_t *q = p;
    p += n;
    return q;
  }
};

inline void read_nibbles(Reader &r, std::vector<uint8_t> &v, size_t n) {
  const uint8_t *b = r.take((n + 1) / 2);
  v.resize(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = (uint8_t)((b[i / 2] >> (4 * (i & 1))) & 15);
}

// decoding table entry of one probability slot
struct Slot {
  uint16_t freq, bias; // bias = slot - cumulative start of the symbol
  uint16_t sym, pad;
};

struct BitReader {
  const uint8_t *p;
  uint64_t nbits, pos = 0;
  inline uint64_t get(int n) { // n <= 33, MSB first
    if (pos + (uint64_t)n > nbits)
      fail("raw bits");
    uint64_t v = 0;
    for (int k = 0; k < n; ++k, ++pos)
      v = (v << 1) | ((p[pos >> 3] >> (7 - (pos & 7))) & 1);
    return v;
  }
};

// where a tensor's values go: base planes into int64 values, the net's tensors
// directly into the output file bytes (row (g, r) at out + g * gstride + r * C * bytes)
struct Dest {
  int64_t *vals = nullptr;
  uint8_t *out = nullptr;
  size_t gstride = 0;
  int bytes = 0;
};

// decodes one tensor; base planes are decoded recursively (they come first)
inline void decode_tensor(Reader &r, uint32_t G, uint32_t R, uint32_t C, const Dest &dst,
                          bool is_base) {
  const uint8_t flags = r.get<uint8_t>();
  const uint32_t K = r.get<uint8_t>();
  const int lt = r.get<uint8_t>();
  const int pb = r.get<uint8_t>();
  if (r.get<uint32_t>() != G || r.get<uint32_t>() != R || r.get<uint32_t>() != C)
    fail("tensor shape");
  if (flags > 1 || ((flags & 1) && (is_base || G < 2)) || K == 0 || lt > 12 ||
      pb < 8 || pb > 15 || (dst.vals == nullptr) == (dst.out == nullptr))
    fail("tensor options");
  // scale bins and context map (only stored with more than one context)
  std::vector<uint8_t> rowbin, colbin, mapv(256, 0);
  if (K > 1) {
    read_nibbles(r, rowbin, (size_t)G * R);
    read_nibbles(r, colbin, C);
    const uint8_t *m = r.take(256);
    for (int i = 0; i < 256; ++i)
      if ((mapv[i] = m[i]) >= K)
        fail("context map");
  } else {
    rowbin.assign((size_t)G * R, 0);
    colbin.assign(C, 0);
  }
  const uint8_t *map = mapv.data();
  std::vector<int64_t> base;
  if (flags & 1) {
    base.resize((size_t)R * C);
    Dest bd;
    bd.vals = base.data();
    decode_tensor(r, 1, R, C, bd, true);
  }
  // frequency tables -> slot tables
  const uint32_t M = 1u << pb, A = alphabet(lt);
  std::vector<Slot> slots((size_t)K * M);
  for (uint32_t k = 0; k < K; ++k) {
    const uint32_t nused = r.get<uint16_t>();
    uint32_t sym = 0, cum = 0;
    for (uint32_t i = 0; i < nused; ++i) {
      sym += r.varint() + (i ? 1 : 0);
      const uint32_t f = r.varint();
      if (sym >= A || f == 0 || cum + f > M)
        fail("frequency table");
      for (uint32_t s = 0; s < f; ++s)
        slots[(size_t)k * M + cum + s] = {(uint16_t)f, (uint16_t)s, (uint16_t)sym, 0};
      cum += f;
    }
    if (cum != M)
      fail("frequency sum");
  }
  const uint64_t nraw = r.get<uint64_t>();
  BitReader bits{r.take(nraw), nraw * 8};
  // two interleaved 64-bit rANS states (value i uses state i % 2: independent
  // dependency chains), 32-bit renormalisation; the stream is copied into a padded
  // buffer so the branchless refill may read 4 bytes past the end
  const uint64_t nr = r.get<uint64_t>();
  if (nr < 16 || (nr & 3) || nr > (1ull << 32))
    fail("rANS stream");
  const uint8_t *src = r.take(nr); // bounds check before allocating
  std::vector<uint8_t> sbuf(nr + 4, 0);
  std::memcpy(sbuf.data(), src, nr);
  const uint8_t *p = sbuf.data(), *end = p + nr;
  auto rd32 = [](const uint8_t *q) {
    return (uint64_t)q[0] | (uint64_t)q[1] << 8 | (uint64_t)q[2] << 16 | (uint64_t)q[3] << 24;
  };
  uint64_t x = rd32(p) | rd32(p + 4) << 32, x1 = rd32(p + 8) | rd32(p + 12) << 32;
  p += 16;
  const uint32_t mask = M - 1;
  const uint32_t T2 = 2u << lt;
  for (uint32_t row = 0; row < G * R; ++row) {
    const Slot *ctab[16];
    for (int cb = 0; cb < 16; ++cb)
      ctab[cb] = slots.data() + (size_t)map[rowbin[row] * 16 + cb] * M;
    const int64_t *bp = base.empty() ? nullptr : base.data() + (size_t)(row % R) * C;
    const uint32_t g = row / R, rr = row % R;
    int64_t *vout = dst.vals ? dst.vals + (size_t)row * C : nullptr;
    uint8_t *bout = dst.out ? dst.out + g * dst.gstride + (size_t)rr * C * dst.bytes : nullptr;
    for (uint32_t c = 0; c < C; ++c) {
      const Slot s = ctab[colbin[c]][x & mask];
      x = (uint64_t)s.freq * (x >> pb) + s.bias;
      const uint64_t w = rd32(p);
      const bool refill = x < RANS_L; // at most one refill (probability bits <= 31)
      x = refill ? (x << 32) | w : x;
      p += refill ? 4 : 0;
      if (p > end)
        fail("rANS overrun");
      std::swap(x, x1); // next value: the other state
      int64_t v;
      if (s.sym < T2) {
        v = (s.sym & 1) ? -(int64_t)((s.sym + 1) >> 1) : (int64_t)(s.sym >> 1);
      } else {
        const int cls = (int)(s.sym - T2) + lt;
        const int64_t mag = (int64_t)((1ull << cls) | bits.get(cls));
        v = bits.get(1) ? -mag : mag;
      }
      if (bp)
        v += bp[c];
      if (vout) {
        vout[c] = v;
      } else { // little-endian two's complement
        const uint32_t u = (uint32_t)v;
        if (dst.bytes == 1) {
          bout[c] = (uint8_t)u;
        } else if (dst.bytes == 2) {
          bout[2 * c] = (uint8_t)u;
          bout[2 * c + 1] = (uint8_t)(u >> 8);
        } else {
          for (int b = 0; b < 4; ++b)
            bout[4 * c + b] = (uint8_t)(u >> (8 * b));
        }
      }
    }
  }
  if (p != end || x != RANS_L || x1 != RANS_L || bits.pos + 8 <= bits.nbits)
    fail("stream end");
}

struct Arch {
  uint32_t L1, L2, L3, Out, NB;
};

// tensor shapes {G, R, C, element bytes} of Network<L1,L2,L3,Out,NB>
inline void tensor_shapes(const Arch &a, uint32_t shp[8][4]) {
  const uint32_t s[8][4] = {{120, 1, 2 * a.L1, 2}, {1, 1, 2 * a.L1, 2},
                            {a.NB, a.L2, a.L1, 1}, {a.NB, 1, a.L2, 4},
                            {a.NB, a.L3, a.L2, 1}, {a.NB, 1, a.L3, 4},
                            {a.NB, a.Out, a.L3, 1}, {a.NB, 1, a.Out, 4}};
  std::memcpy(shp, s, sizeof(s));
}

// decodes a .dhn blob into the bytes of the original .quant file (verified)
inline std::vector<uint8_t> unpack(const unsigned char *data, size_t size) {
  Reader r{data, data + size};
  if (size < 4 || std::memcmp(data, "DHN4", 4) != 0)
    fail("magic");
  r.p += 4;
  const uint64_t out_size = r.get<uint64_t>();
  const uint64_t sum = r.get<uint64_t>();
  Arch a;
  a.L1 = r.get<uint32_t>();
  a.L2 = r.get<uint32_t>();
  a.L3 = r.get<uint32_t>();
  a.Out = r.get<uint32_t>();
  a.NB = r.get<uint32_t>();
  if (a.L1 == 0 || a.L1 > 65536 || a.L2 > 4096 || a.L3 > 4096 || a.Out > 4096 || a.NB > 4096)
    fail("architecture");
  uint32_t shp[8][4];
  tensor_shapes(a, shp);
  // file layout: acc weights, acc biases, then per layer and bucket weights + biases
  size_t total = 0;
  for (int k = 0; k < 8; ++k)
    total += (size_t)shp[k][0] * shp[k][1] * shp[k][2] * shp[k][3];
  if (total != out_size || total > (1ull << 30))
    fail("size");
  std::vector<uint8_t> out(total);
  Dest d[8];
  size_t off = 0;
  for (int k = 0; k < 2; ++k) {
    d[k].out = out.data() + off;
    d[k].gstride = (size_t)shp[k][1] * shp[k][2] * shp[k][3];
    d[k].bytes = (int)shp[k][3];
    off += (size_t)shp[k][0] * d[k].gstride;
  }
  for (int l = 2; l < 8; l += 2) {
    const size_t nw = (size_t)shp[l][1] * shp[l][2] * shp[l][3];
    const size_t nb = (size_t)shp[l + 1][2] * shp[l + 1][3];
    d[l].out = out.data() + off;
    d[l + 1].out = out.data() + off + nw;
    d[l].gstride = d[l + 1].gstride = nw + nb;
    d[l].bytes = (int)shp[l][3];
    d[l + 1].bytes = (int)shp[l + 1][3];
    off += (size_t)a.NB * (nw + nb);
  }
  for (int k = 0; k < 8; ++k)
    decode_tensor(r, shp[k][0], shp[k][1], shp[k][2], d[k], false);
  if (r.p != r.end)
    fail("trailing bytes");
  if (fnv64(out.data(), out.size()) != sum)
    fail("checksum");
  return out;
}

// loads a net from an embedded .dhn blob
template <typename Net>
void load_packed(Net &net, const unsigned char *data, size_t size) {
  const std::vector<uint8_t> raw = unpack(data, size);
  net.load_from_array(raw.data(), raw.size());
}

} // namespace netpack

#endif // NETPACK_H
