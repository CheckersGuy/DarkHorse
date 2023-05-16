//
// Created by Robin on 19.01.2018.
//

#ifndef CHECKERENGINEX_MOVEPICKER_H
#define CHECKERENGINEX_MOVEPICKER_H

#include "Move.h"
#include "MoveListe.h"
#include "Network.h"
#include "Position.h"
#include "types.h"
namespace Statistics {
class MovePicker {
private:
  std::array<int, 32 * 16> history{0};

public:
  std::array<std::array<Move, MAX_KILLERS>, MAX_PLY> killer_moves;
  std::array<std::array<int, 32 * 16>, 32 * 16> counter_history;
  std::array<std::array<int, 32 * 16>, 32 * 16> follow_history;
  int get_move_score(Position pos, Move move, Move previous, Move previous_own,
                     Depth depth);

  int get_history_index(Position pos, Move move);

  int get_move_score(Position current, Depth depth, int ply, Move move,
                     Move previous, Move previous_own, Move ttMove);

  void clear_scores();

  void reduce_scores();

  void update_scores(Position pos, Move *list, Move move, Move previous,
                     Move previous_own, int depth);

  static int get_move_encoding(Move move);
  static int get_policy_encoding(Color color, Move move);
  void init();
};

extern MovePicker mPicker;

} // namespace Statistics
#endif // CHECKERENGINEX_MOVEPICKER_H
