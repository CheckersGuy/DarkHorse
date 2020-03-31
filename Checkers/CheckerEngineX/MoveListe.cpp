#include "MoveListe.h"
#include "Weights.h"


void MoveListe::sort(Move ttMove, bool inPVLine, Color color) {
    for (auto i = 0; i < liste.size(); ++i) {
        scores[i] = Statistics::mPicker.getMoveScore(liste[i], color, ttMove);
    }
    auto start_index = (inPVLine) ? 1 : 0;
    help_sort(start_index);
}

void MoveListe::help_sort(int start_index) {
    for (int i = start_index + 1; i < moveCounter; i++) {
        int tmp = scores[i];
        Move tmpMove = liste[i];
        int j;
        for (j = i; j > (start_index) && scores[j - 1] < tmp; j--) {
            liste[j] = liste[j - 1];
            scores[j] = scores[j - 1];
        }
        liste[j] = tmpMove;
        scores[j] = tmp;
    }
}

void MoveListe::putFront(const Move &move) {
    auto it = std::find(liste.begin(), liste.end(), move);
    if (it != liste.end()) {
        std::swap(*liste.begin(), *it);
    }
}

MoveListe &MoveListe::operator=(const MoveListe &other) {
    this->moveCounter = other.moveCounter;
    std::copy(other.liste.begin(), other.liste.begin() + other.moveCounter, liste.begin());
    return *this;
}

void MoveListe::sort_static(Color mover, const Position &pos, const Move &ttMove) {
    for (auto i = 0; i < moveCounter; ++i) {
        Position copy = pos;
        copy.makeMove(liste[i]);
        Board board;
        board = copy;

        auto eval = Statistics::mPicker.getMoveScore(liste[i], mover, ttMove);
        Line local;
        auto score = quiescene<NONPV>(board, -INFINITE, INFINITE, local, 0) / scalfac;

        eval += score;

        scores[i] = eval;
    }
    auto start_index = 1;
    help_sort(start_index);

}