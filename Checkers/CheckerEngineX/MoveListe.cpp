#include "MoveListe.h"
#include "MovePicker.h"
#include <cstdint>
#include <limits>

void MoveListe::reset() { moveCounter = 0; }

extern Line mainPV;

MoveListe &MoveListe::operator=(const MoveListe &other) {
  for (auto i = 0; i < other.moveCounter; ++i) {
    liste[i] = other.liste[i];
  }
  this->moveCounter = other.moveCounter;
  return *this;
}

void MoveListe::move_to_front(int start_index, Move move) {
  for (auto i = start_index; i < moveCounter; ++i) {
    if (liste[i] == move) {
      const Move temp = liste[i];
      liste[i] = liste[start_index];
      liste[start_index] = temp;
      break;
    }
  }
}
