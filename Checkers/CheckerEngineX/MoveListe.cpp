#include "MoveListe.h"
#include "Weights.h"

void MoveListe::reset() {
    moveCounter = 0;
}

extern Line mainPV;

void MoveListe::sort(Position current,Depth depth,Move ttMove, bool inPVLine, Color color) {

    if (moveCounter <= 1)
        return;

    const int start_index = (inPVLine) ? 1 : 0;


    for (auto i = 0; i < moveCounter; ++i) {
        Move m = liste[i];
        scores[i] = (short) Statistics::mPicker.getMoveScore(current,depth,m, color, ttMove);
    }


    for (int i = start_index + 1; i < moveCounter; ++i) {
        int tmp = scores[i];
        Move tmpMove = liste[i];
        int j;
        for (j = i; j > (start_index) && scores[j - 1] < tmp; --j) {
            liste[j] = liste[j - 1];
            scores[j] = scores[j - 1];
        }
        liste[j] = tmpMove;
        scores[j] = (short) tmp;
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

uint8_t MoveListe::get_move_index(Move move) const {
    for (uint32_t i = 0u; i < moveCounter; ++i) {
        if (move == liste[i])
            return i;
    }
    return Move_Index_None;
}

