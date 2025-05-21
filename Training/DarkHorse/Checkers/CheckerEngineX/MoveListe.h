#ifndef CHECKERSTEST_MOVELISTE_H
#define CHECKERSTEST_MOVELISTE_H

#include "Line.h"
#include "Move.h"
#include "Position.h"
#include "types.h"
#include <array>

struct Local {
  Move previous, previous_own;
};

class MoveListe {

private:
  int moveCounter = 0;

public:
  std::array<Move, 40> liste;

  int length() const;

  void add_move(Move next);

  template <typename Oracle>
  void sort(Position current, Depth depth, Ply ply, Move ttMove,
            int start_index, Oracle func);

  bool is_empty() const;

  const Move &operator[](int index) const;

  Move &operator[](int index);

  void reset();

  void move_to_front(int start_index, Move move);

  template <typename Oracle> void move_to_front(int start_index, Oracle func);

  auto begin() { return liste.begin(); }

  auto end() {
    auto it = liste.begin();
    std::advance(it, moveCounter);
    return it;
  }

  MoveListe &operator=(const MoveListe &other);

  template <MoveType type> inline void visit(uint32_t &maske, uint32_t &next) {
    add_move(Move{maske, next});
  };

  template <MoveType type>
  inline void visit(uint32_t &from, uint32_t &to, uint32_t &captures) {
    add_move(Move{from, to, captures});
  };
};

inline bool MoveListe::is_empty() const { return this->moveCounter == 0; }

inline const Move &MoveListe::operator[](int index) const {
  return liste[index];
}

inline Move &MoveListe::operator[](int index) { return liste[index]; }

inline void MoveListe::add_move(Move next) {
  liste[this->moveCounter++] = next;
}

inline int MoveListe::length() const { return moveCounter; }

template <typename Oracle>
void MoveListe::sort(Position current, Depth depth, Ply ply, Move ttMove,
                     int start_index, Oracle oracle) {

  if (moveCounter - start_index <= 1)
    return;
  std::array<int, 40> scores;

  for (auto i = start_index; i < moveCounter; ++i) {
    Move m = liste[i];
    scores[i] = oracle(m);
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

template <typename Oracle>
void MoveListe::move_to_front(int start_index, Oracle func) {
  // finds the best move according to our oracle and swaps it to the front

  int max_score = -10000;
  int max_index = 0;

  for (auto i = start_index; i < moveCounter; ++i) {
    const auto score = func(liste[i]);
    if (score > max_score) {
      max_score = score;
      max_index = i;
    }
  }
  // swap moves to front
  const Move temp = liste[start_index];
  liste[start_index] = liste[max_index];
  liste[max_index] = temp;
}

#endif // CHECKERSTEST_MOVELISTE_H
