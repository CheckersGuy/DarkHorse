#ifndef TRAINING_SPRT_H
#define TRAINING_SPRT_H

// Pentanomial statistics and SPRT for engine matches.
//
// Every opening is played twice with colours swapped; the two results form a
// pair. A pair score (from engine 1's view) is 0, 0.5, 1, 1.5 or 2 points,
// counted in five bins: LL, LD, DD+WL, WD, WW. Working with pairs removes most
// of the opening noise from the variance.
//
// The LLR is the exact pentanomial GSPRT (as in fishtest) with logistic Elo
// bounds elo0/elo1; see Pentanomial::llr.

#include <array>
#include <cmath>
#include <cstdint>

namespace Sprt {

inline double elo_to_score(double elo) {
  return 1.0 / (1.0 + std::pow(10.0, -elo / 400.0));
}

inline double score_to_elo(double s) {
  s = std::fmin(std::fmax(s, 1e-9), 1.0 - 1e-9);
  return -400.0 * std::log10(1.0 / s - 1.0);
}

struct Pentanomial {
  // index = pair points in half points: 0 = LL, 1 = LD, 2 = DD/WL, 3 = WD, 4 = WW
  std::array<uint64_t, 5> n{};

  void add(int half_points) { n[half_points]++; }

  uint64_t pairs() const {
    uint64_t s = 0;
    for (auto c : n)
      s += c;
    return s;
  }

  // mean and variance of the pair score as a fraction (0, .25, .5, .75, 1).
  // Empty bins get a tiny pseudo-count so the variance is never zero early on.
  void mean_var(double &mean, double &var) const {
    bool any_empty = false;
    for (auto c : n)
      any_empty |= (c == 0);
    const double reg = any_empty ? 1e-3 : 0.0;

    double total = 0.0;
    for (auto c : n)
      total += (double)c + reg;
    mean = 0.0;
    for (int i = 0; i < 5; ++i)
      mean += (n[i] + reg) / total * (i / 4.0);
    var = 0.0;
    for (int i = 0; i < 5; ++i) {
      const double d = i / 4.0 - mean;
      var += (n[i] + reg) / total * d * d;
    }
  }

  // regularised bin frequencies (see mean_var)
  std::array<double, 5> freqs() const {
    bool any_empty = false;
    for (auto c : n)
      any_empty |= (c == 0);
    const double reg = any_empty ? 1e-3 : 0.0;
    double total = 0.0;
    for (auto c : n)
      total += (double)c + reg;
    std::array<double, 5> p{};
    for (int i = 0; i < 5; ++i)
      p[i] = (n[i] + reg) / total;
    return p;
  }

  // Maximum-likelihood pentanomial distribution with mean score s:
  // q_i = p_i / (1 + t (x_i - s)), with t chosen so that sum q_i x_i = s.
  static std::array<double, 5> constrained_mle(const std::array<double, 5> &p,
                                               double s) {
    auto f = [&](double t) {
      double v = 0.0;
      for (int i = 0; i < 5; ++i) {
        const double d = i / 4.0 - s;
        v += p[i] * d / (1.0 + t * d);
      }
      return v; // decreasing in t
    };
    // 1 + t (x - s) > 0 for all x in [0, 1]
    double lo = -1.0 / (1.0 - s) + 1e-12, hi = 1.0 / s - 1e-12;
    for (int it = 0; it < 200; ++it) {
      const double mid = 0.5 * (lo + hi);
      (f(mid) > 0.0 ? lo : hi) = mid;
    }
    std::array<double, 5> q{};
    double sum = 0.0;
    for (int i = 0; i < 5; ++i) {
      q[i] = p[i] / (1.0 + lo * (i / 4.0 - s));
      sum += q[i];
    }
    for (auto &v : q)
      v /= sum;
    return q;
  }

  // Exact GSPRT log-likelihood ratio (as in fishtest): the likelihood of the
  // observed pairs under the best pentanomial model with Elo elo1 vs elo0.
  // Unlike the normal approximation N/2 (s1-s0)(2s-s0-s1)/var this stays
  // sane when the variance is tiny, e.g. after a run of draw-draw pairs.
  double llr(double elo0, double elo1) const {
    const uint64_t N = pairs();
    if (N == 0)
      return 0.0;
    const auto p = freqs();
    const auto q0 = constrained_mle(p, elo_to_score(elo0));
    const auto q1 = constrained_mle(p, elo_to_score(elo1));
    double v = 0.0;
    for (int i = 0; i < 5; ++i)
      v += p[i] * std::log(q1[i] / q0[i]);
    return N * v;
  }

  // Elo estimate with a 95% confidence half-width
  void elo(double &elo, double &error95) const {
    const uint64_t N = pairs();
    if (N == 0) {
      elo = 0.0;
      error95 = 0.0;
      return;
    }
    double s, var;
    mean_var(s, var);
    const double se = std::sqrt(var / N);
    elo = score_to_elo(s);
    error95 =
        (score_to_elo(s + 1.959964 * se) - score_to_elo(s - 1.959964 * se)) /
        2.0;
  }
};

inline double lower_bound(double alpha, double beta) {
  return std::log(beta / (1.0 - alpha));
}

inline double upper_bound(double alpha, double beta) {
  return std::log((1.0 - beta) / alpha);
}

} // namespace Sprt

#endif // TRAINING_SPRT_H
