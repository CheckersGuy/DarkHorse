//
// Created by Robin on 04.01.2018.
//

#include "MoveListe.h"


void MoveListe::sort(Move ttMove, bool inPVLine, Color color) {

    for (int i = (inPVLine) ? 1 : 0; i < moveCounter; ++i) {
        scores[i] = Statistics::mPicker.getMoveScore(liste[i], color, ttMove);
    }

    int j;
    for (int i = (inPVLine) ? 2 : 1; i < moveCounter; i++) {
        int tmp = scores[i];
        Move tmpMove = liste[i];
        for (j = i; j > ((inPVLine) ? 1 : 0) && scores[j - 1] < tmp; j--) {
            liste[j] = liste[j - 1];
            scores[j] = scores[j - 1];
        }
        liste[j] = tmpMove;
        scores[j] = tmp;
    }


}
MoveListIterator MoveListe::begin() {
    MoveListIterator iter;
    iter.p = &liste[0];
    return iter;
}


MoveListIterator MoveListe::end() {
    MoveListIterator iter;
    iter.p = &liste[moveCounter];
    return iter;
}