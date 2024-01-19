#include "MoveListe.h"
#include "MovePicker.h"
#include <cstdint>
#include <limits>

void MoveListe::reset() { moveCounter = 0; }

extern Line mainPV;

void MoveListe::sort(Position current, Depth depth, Ply ply, Move ttMove,
                     int start_index) {

  if (moveCounter - start_index <= 1)
    return;
  std::array<int, 40> scores;

  for (auto i = start_index; i < moveCounter; ++i) {
    Move m = liste[i];
    scores[i] =
        Statistics::mPicker.get_move_score(current, depth, ply, m, ttMove);
  }

  for (int i = start_index + 1; i < moveCounter; ++i) {
    const int tmp = scores[i];
    Move tmpMove = liste[i];
    int j;
    for (j = i; j > (start_index) && scores[j - 1] < tmp; --j) {
      liste[j] = liste[j - 1];
      scores[j] = scores[j - 1];
    }
    liste[j] = tmpMove;
    scores[j] = tmp;
  }
}
void MoveListe::remove(Move move) {

  int i;
  for (i = 0; i < moveCounter; ++i) {
    if (liste[i] == move)
      break;
  }

  for (int k = i + 1; k < moveCounter; ++k) {
    liste[k - 1] = liste[k];
  }

  moveCounter = moveCounter - 1;
}

bool MoveListe::put_front(Move other) {
  auto it = std::find(begin(), end(), other);
  if (it != end()) {
    std::swap(liste[0], *it);
    return true;
  }
  return false;
}

MoveListe &MoveListe::operator=(const MoveListe &other) {
  for (auto i = 0; i < other.moveCounter; ++i) {
    liste[i] = other.liste[i];
  }
  this->moveCounter = other.moveCounter;
  return *this;
}
