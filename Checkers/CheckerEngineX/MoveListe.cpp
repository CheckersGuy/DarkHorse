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

    int start_index = (inPVLine) ? 1 : 0;

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
    //more elegant solution since putFront is rarely ever used

    if (moveCounter <= 1)
        return;

    auto it = std::find(liste.begin(), liste.end(), move);
    if (it != liste.end()) {
        std::swap(*liste.begin(), *it);
    }
}

MoveListe &MoveListe::operator=(const MoveListe &other) {
    std::copy(other.liste.begin(), other.liste.begin() + other.moveCounter, liste.begin());
    this->moveCounter = other.moveCounter;
    return *this;
}

std::optional<uint8_t> MoveListe::get_move_index(Move move) const {
    for (uint8_t i = 0; i < moveCounter; ++i) {
        if (move == liste[i])
            return std::make_optional(i);
    }
    return std::nullopt;
}

void MoveListe::sort_static(Color mover, const Position &pos, const Move &ttMove) {


}