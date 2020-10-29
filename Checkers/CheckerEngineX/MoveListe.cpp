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

/*    auto result = get_move_index(move);
    if (result.has_value()) {
        const int tmp_score = scores[0];
        liste[result.value()] = liste[0];
        liste[0] = move;
        scores[0] = scores[result.value()];
        scores[result.value()] = tmp_score;
    }*/
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

std::optional<uint8_t> MoveListe::get_move_index(Move move) const {
    for (auto i = 0; i < moveCounter; ++i) {
        if (move == liste[i])
            return std::make_optional(i);
    }
    return std::nullopt;
}

void MoveListe::sort_static(Color mover, const Position &pos, const Move &ttMove) {


}