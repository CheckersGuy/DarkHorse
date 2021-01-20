#include "MoveListe.h"
#include "Weights.h"

void MoveListe::reset() {
    moveCounter = 0;
}

void MoveListe::sort(Move ttMove, bool inPVLine, Color color) {

    if (moveCounter <= 1)
        return;

    for (auto i = 0; i < moveCounter; ++i) {
        scores[i] = Statistics::mPicker.getMoveScore(liste[i], color, ttMove);
    }

    const int start_index = (inPVLine) ? 1 : 0;

    for (int i = start_index + 1; i < moveCounter; ++i) {
        int tmp = scores[i];
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

void MoveListe::putFront(const Move move) {

    if (moveCounter <= 1)
        return;

    auto it = std::find(begin(), end(), move);
    if (it != end()) {
        std::swap(*begin(), *it);
    }


}

MoveListe &MoveListe::operator=(const MoveListe &other) {
    std::copy(begin(), end(), begin());
    this->moveCounter = other.moveCounter;
    return *this;
}

uint32_t MoveListe::get_move_index(Move move) const {
    for (uint32_t i = 0u; i < moveCounter; ++i) {
        if (move == liste[i])
            return i;
    }
    return Move_Index_None;
}
