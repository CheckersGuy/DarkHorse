//
// Created by Robin on 19.01.2018.
//

#include "MovePicker.h"
#include "Position.h"
#include <assert.h>
namespace Statistics {

MovePicker mPicker;

void MovePicker::init() {}

int MovePicker::get_move_encoding(Move move) {
  int dir;

  if ((((move.from & MASK_L3) << 3) == move.to) ||
      (((move.from & MASK_L5) << 5) == move.to)) {
    dir = 0;

  } else if (((move.from) << 4) == move.to) {
    dir = 1;

  } else if (((move.from) >> 4) == move.to) {
    dir = 2;

  } else if ((((move.from & MASK_R3) >> 3) == move.to) ||
             (((move.from & MASK_R5) >> 5) == move.to)) {
    dir = 3;
  };

  return 4 * move.get_from_index() + dir;
}

int MovePicker::get_policy_encoding(Color mover, Move move) {
  if (mover == BLACK) {
    Move temp;
    temp.from = getMirrored(move.from);
    temp.to = getMirrored(move.to);
    temp.captures = getMirrored(move.captures);
    return get_move_encoding(temp);
  }
  return get_move_encoding(move);
}

int MovePicker::get_history_index(Position pos, Move move) {
  int t;

  if ((move.from & (pos.BP & pos.K)) != 0) {
    t = 0;
  } else if ((move.from & (pos.WP & pos.K)) != 0) {
    t = 1;
  } else if ((move.from & pos.BP) != 0) {
    t = 2;
  } else if ((move.from & pos.WP) != 0) {
    t = 3;
  } else {
    t = 0;
  }

  int orig_sq = move.get_from_index();
  int dir = 0;

  if ((((move.from & MASK_L3) << 3) == move.to) ||
      (((move.from & MASK_L5) << 5) == move.to)) {
    dir = 0;

  } else if (((move.from) << 4) == move.to) {
    dir = 1;

  } else if (((move.from) >> 4) == move.to) {
    dir = 2;

  } else if ((((move.from & MASK_R3) >> 3) == move.to) ||
             (((move.from & MASK_R5) >> 5) == move.to)) {
    dir = 3;
  };

  const int index = 16 * orig_sq + 4 * dir + t;
  return index;
}

void MovePicker::clear_scores() {
  std::fill(history.begin(), history.end(), 0);
  for (auto i = 0; i < killer_moves.size(); ++i) {
    for (auto k = 0; k < MAX_KILLERS; ++k) {
      killer_moves[i][k] = Move{};
    }
  }
  for (auto i = 0; i < counter_history.size(); ++i) {
    for (auto k = 0; k < counter_history[0].size(); ++k) {
      counter_history[i][k] = 0;
    }
  }

  for (auto i = 0; i < follow_history.size(); ++i) {
    for (auto k = 0; k < follow_history[0].size(); ++k) {
      follow_history[i][k] = 0;
    }
  }
}

int MovePicker::get_move_score(Position pos, Move move, Move previous,
                               Move previous_own, Depth depth) {
  // const int index = get_move_encoding(pos.get_color(),move);
  const int index = get_history_index(pos, move);
  int score = history[index];

  if (!previous.is_capture() && !move.is_capture() && !previous.is_empty()) {
    auto counter = counter_history[get_history_index(pos, previous)][index];
    score += counter;
  }

  if (!previous_own.is_empty() && !previous_own.is_capture() &&
      !move.is_capture()) {
    auto follow = follow_history[get_history_index(pos, previous_own)][index];
    score += follow;
  }

  return score;
}

int MovePicker::get_move_score(Position current, Depth depth, int ply,
                               Move move, Move previous, Move previous_own,
                               Move ttMove) {
  if (move == ttMove) {
    return std::numeric_limits<int32_t>::max();
  }
  if (move.is_capture()) {
    const uint32_t kings_captured = move.captures & current.K;
    const uint32_t pawns_captured = move.captures & (~current.K);
    return (int)(Bits::pop_count(kings_captured) * 3 +
                 Bits::pop_count(pawns_captured) * 2);
  }

  for (auto i = 0; i < MAX_KILLERS; ++i) {
    if (move == killer_moves[ply][i]) {
      return std::numeric_limits<int32_t>::max() - 1000;
    }
  }

  return get_move_score(current, move, previous, previous_own, depth);
}

void update_history_score(int &score, int delta) { score += delta; }

void MovePicker::update_scores(Position pos, Move *liste, Move move,
                               Move previous, Move previous_own, int depth) {
  const int index = get_history_index(pos, move);
  const int delta = std::min(depth * depth, 18 * 18);
  update_history_score(history[index], delta);
  Move top = liste[0];

  if (!previous.is_capture() && !move.is_capture() && !previous.is_empty()) {
    counter_history[get_history_index(pos, previous)]
                   [get_history_index(pos, move)] += delta;
  }

  if (!previous_own.is_empty() && !previous_own.is_capture() &&
      !move.is_capture()) {
    follow_history[get_history_index(pos, previous_own)]
                  [get_history_index(pos, move)] += delta;
  }

  while ((top = *(liste++)) != move) {
    int &score = history[get_history_index(pos, top)];
    update_history_score(score, -delta);
    if (!previous.is_capture() && !top.is_capture() && !previous.is_empty()) {
      counter_history[get_history_index(pos, previous)]
                     [get_history_index(pos, top)] -= delta;
    }
    if (!previous_own.is_empty() && !previous_own.is_capture() &&
        !top.is_capture()) {
      follow_history[get_history_index(pos, previous_own)]
                    [get_history_index(pos, top)] -= delta;
    }
  }
}
} // namespace Statistics
