//
// Created by Robin on 04.01.2018.
//

#include "MoveListe.h"


void MoveListe::sort(Move ttMove, bool inPVLine, Color color) {
    std::sort(begin() + inPVLine, end(), [&](Move one, Move two){
        return Statistics::mPicker.getMoveScore(one, color, ttMove) > Statistics::mPicker.getMoveScore(two, color, ttMove);
    });

}

void MoveListe::putFront(const Move &move) {
    auto it = std::find(begin(), end(), move);
    if (it != end()) {
        std::swap(liste[0], *it);
    }
}

