#define CHECKERBOARD
#include "GameLogic.h"
#include <iostream>
#include <stdint.h>
// Everything related to the checkerboard interface
#define CB_WHITE 1
#define CB_BLACK 2
#define CB_MAN 4
#define CB_KING 8
#define CB_FREE 0

#define CB_DRAW 0
#define CB_WIN 1
#define CB_LOSS 2
#define CB_UNKNOWN 3

#define GT_ENGLISH 21
#define GT_ITALIAN 22
#define GT_SPANISH 24
#define GT_RUSSIAN 25
#define GT_BRAZILIAN 26
#define GT_CZECH 29

/* enginecommand "get book", and "set book" values. */
#define CB_BOOK_NONE 0
#define CB_BOOK_ALL_KINDS_MOVES 1
#define CB_BOOK_GOOD_MOVES 2
#define CB_BOOK_BEST_MOVES 3

/* getmove() 'info' argument bit definitions. */
#define CB_RESET_MOVES 1
#define CB_EXACT_TIME 2

struct coor {
  int x;
  int y;
};

struct CBmove {
  int jumps;
  int newpiece;
  int oldpiece;
  struct coor from, to;
  struct coor path[12]; // intermediate squares to jump to
  struct coor del[12];  // squares where men are removed
  int delpiece[12];     // piece type which is removed
};

template <size_t size, typename Generator>
constexpr auto get_lut1(Generator &&generator) {
  using Type = decltype(generator(0));
  std::array<Type, size> result;

  for (auto i = 0; i < size; ++i) {
    result[i] = generator(i);
  }
  return result;
};

Position last_position;
bool engine_initialized = false;
constexpr std::array<size_t, 32> To64 = get_lut1<32>([](size_t index) {
  auto row = index / 4;
  return ((row & 1) == 0) ? 2 * index + 1 : 2 * index;
});
Board game_board;

extern "C" int getmove(int board[8][8], int color, double maxtime,
                       char str[1024], int *playnow, int info, int moreinfo,
                       struct CBmove *move);

// extern"C" unsigned int CB_GETGAMETYPE();	/* return GT_ENGLISH,
// GT_ITALIAN, ... */
//  INT (WINAPI *CB_ISLEGAL)(Board8x8 board, int color, int from, int to, CBmove
//  *move);
extern "C" int enginecommand(char command[256], char reply[1024]);

inline void numbertocoors(int number, int *x, int *y, int gametype) {
  switch (gametype) {
  case GT_ITALIAN:
    number--;                // number e 0...31
    *y = number / 4;         // *y e 0...7
    *x = 2 * ((number % 4)); // *x e {0,2,4,6}
    if (((*y) % 2))          // adjust x on odd rows
      (*x)++;
    break;

  case GT_SPANISH:
    number--;
    *y = number / 4;
    *y = 7 - *y;
    *x = 2 * (3 - (number % 4)); // *x e {0,2,4,6}
    if (((*y) % 2))              // adjust x on odd rows
      (*x)++;
    break;

  case GT_CZECH: // TODO: check that this is correct!
    number--;    // number e 0...31
    number = 33 - number;
    *y = number / 4;         // *y e 0...7
    *x = 2 * ((number % 4)); // *x e {0,2,4,6}
    if (((*y) % 2))          // adjust x on odd rows
      (*x)++;
    break;

  default:
    number--;
    *y = number / 4;
    *x = 2 * (3 - number % 4);
    if ((*y) % 2)
      (*x)++;
  }
}

inline void numbertocoors(int number, coor *c, int gametype) {
  numbertocoors(number, &c->x, &c->y, gametype);
}

/*
 * Give the x,y coordinates for a Board8x8, return the square number (1..32).
 */
inline int coorstonumber(int x, int y, int gametype) {
  int number;

  switch (gametype) {
  case GT_ITALIAN:
    // italian rules
    number = 1;
    number += 4 * y;
    number += x / 2;
    break;

  case GT_SPANISH:
    // spanish rules
    number = 1;
    number += 4 * (7 - y);
    number += (7 - x) / 2;
    break;

  case GT_CZECH:
    // TODO: make sure this is correct for czech rules
    number = 1;
    number += 4 * y;
    number += x / 2;
    number = 33 - number;
    break;

  default:
    number = 0;
    number += 4 * (y + 1);
    number -= x / 2;
  }

  return number;
}
