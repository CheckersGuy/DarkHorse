
//
// Created by Robin on 14.01.2018.
//

#ifndef CHECKERENGINEX_PERFT_H
#define CHECKERENGINEX_PERFT_H

#include "MGenerator.h"
#include "Position.h"
#include <atomic>
#include <cstring>
#include <deque>
#include <optional>
#include <vector>
namespace Perft {

uint64_t perft_check(Board &board, int depth);

} // namespace Perft
#endif // CHECKERENGINEX_PERFT_H
