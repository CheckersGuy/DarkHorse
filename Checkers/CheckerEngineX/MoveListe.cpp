#include "MoveListe.h"
#include "Weights.h"

void MoveListe::reset() {
    moveCounter = 0;
}

extern Line mainPV;

void MoveListe::sort(Position current, Depth depth, int ply, Move ttMove, int start_index) {

    if (moveCounter - start_index <= 1)
        return;

   //Statistics::mPicker.policy.compute_incre_forward_pass(current);
    for (auto i = start_index; i < moveCounter; ++i) {
        Move m = liste[i];
        scores[i] = (short) Statistics::mPicker.get_move_score(current, depth, ply, m, ttMove);
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

void MoveListe::put_front(Move other) {

    if (moveCounter <= 1)
        return;

    auto it = std::find(begin(), end(), other);
    if (it != end()) {
        std::swap(*begin(), *it);
    }
}

void MoveListe::put_front(int start_index,int move_index){
    const Move temp = liste[start_index];
    liste[start_index]=liste[move_index];
    liste[move_index]=temp;
}

MoveListe &MoveListe::operator=(const MoveListe &other) {
    for (auto i = 0; i < other.moveCounter; ++i) {
        liste[i] = other.liste[i];
    }
    this->moveCounter = other.moveCounter;
    return *this;
}



