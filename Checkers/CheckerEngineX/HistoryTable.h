#include "Move.h"

struct HistoryEntry {
  int score{0};    // accumulated history score
  int visits{0};   // number of times this move caused a cutoff
  int attempts{0}; // number of times this move was tried
};

struct HistoryTable {
private:
  // indexed by [color][from_index][to_index]
  // color: 0 = BLACK, 1 = WHITE
  std::array<std::array<std::array<HistoryEntry, 32>, 32>, 2> table{};

  // track maximum score for normalization
  int max_score{1};

  // controls how quickly history trust builds up
  // higher = need more visits before trusting history over policy
  static constexpr int PRIOR_WEIGHT = 20;

  // gravity constant to prevent overflow
  static constexpr int MAX_HISTORY = 32768;

public:
  void clear() {
    for (auto &color_table : table) {
      for (auto &from_table : color_table) {
        for (auto &entry : from_table) {
          entry = HistoryEntry{};
        }
      }
    }
    max_score = 1;
  }

  // called when move causes a beta cutoff (good move)
  void update_good(Move move, Color color, Depth depth) {
    auto &entry = get_entry(move, color);
    auto bonus = depth * depth;
    // gravity formula keeps score bounded
    entry.score += bonus - entry.score * bonus / MAX_HISTORY;
    entry.visits++;
    entry.attempts++;
    // track maximum for normalization
    max_score = std::max(max_score, std::abs(entry.score));
  }

  // called when move failed to cause cutoff (bad move)
  void update_bad(Move move, Color color, Depth depth) {
    auto &entry = get_entry(move, color);
    auto penalty = depth * depth;
    entry.score -= penalty - entry.score * penalty / MAX_HISTORY;
    entry.attempts++;
    max_score = std::max(max_score, std::abs(entry.score));
  }

  // returns weight for history [0,1]
  // approaches 1 as visits increases
  float get_history_weight(Move move, Color color) const {
    const auto &entry = get_entry(move, color);
    return static_cast<float>(entry.visits) /
           static_cast<float>(entry.visits + PRIOR_WEIGHT);
  }

  // returns normalized history score in [-1, 1]
  float get_normalized_score(Move move, Color color) const {
    const auto &entry = get_entry(move, color);
    return static_cast<float>(entry.score) / static_cast<float>(max_score);
  }

  int get_visits(Move move, Color color) const {
    return get_entry(move, color).visits;
  }

  int get_attempts(Move move, Color color) const {
    return get_entry(move, color).attempts;
  }

private:
  HistoryEntry &get_entry(Move move, Color color) {
    const int color_idx = (color == WHITE) ? 1 : 0;
    return table[color_idx][move.get_from_index()][move.get_to_index()];
  }

  const HistoryEntry &get_entry(Move move, Color color) const {
    const int color_idx = (color == WHITE) ? 1 : 0;
    return table[color_idx][move.get_from_index()][move.get_to_index()];
  }
};

struct PolicyHistoryCombiner {
private:
  const HistoryTable &history;

  // controls the influence of history vs policy
  // higher = history has more influence when visits are high
  static constexpr float HISTORY_INFLUENCE = 0.5f;

public:
  explicit PolicyHistoryCombiner(const HistoryTable &hist) : history(hist) {}

  // combines policy prior with history score
  // hist_weight approaches HISTORY_INFLUENCE as visits increase
  float combine(Move move, Color color, float policy_score) const {
    float hist_weight =
        history.get_history_weight(move, color) * HISTORY_INFLUENCE;
    float policy_weight = 1.0f - hist_weight;
    float hist_score = history.get_normalized_score(move, color);

    return policy_weight * policy_score + hist_weight * hist_score;
  }

  // returns combined score scaled to int for compatibility
  // with existing oracle interface
  int combine_scaled(Move move, Color color, int policy_score,
                     int scale) const {
    float normalized_policy =
        static_cast<float>(policy_score) / static_cast<float>(scale);
    float combined = combine(move, color, normalized_policy);
    return static_cast<int>(combined * scale);
  }
};
