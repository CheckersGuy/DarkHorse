#include "MoveListe.h"
#include "MovePicker.h"
#include <cstdint>
#include <limits>

void MoveListe::reset() { moveCounter = 0; }

extern Line mainPV;

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
