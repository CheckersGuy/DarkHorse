#include "MoveListe.h"
#include "MovePicker.h"

void MoveListe::reset() {
    moveCounter = 0;
}

extern Line mainPV;

void MoveListe::sort(Position current, Local& local, Move ttMove, int start_index) {

    if (moveCounter - start_index <= 1)
        return;
      std::array<int,40> scores;

    Statistics::mPicker.policy.compute_incre_forward_pass(current);
    for (auto i = start_index; i < moveCounter; ++i) {
        Move m = liste[i];
        scores[i] =Statistics::mPicker.get_move_score(current, local.depth, local.ply, m,local.previous, local.previous_own, ttMove);
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
        scores[j] = tmp;
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

bool MoveListe::put_front(Move other) {
	return put_front(0,other);
}

bool MoveListe::put_front(int start_index, Move other) {
    if (moveCounter<= 1)
			return false;

        auto tmp = liste[start_index];
        for(auto i=start_index; i<moveCounter; ++i)
        {
            if(liste[i]==other) {
                liste[start_index]=other;
                liste[i]=tmp;
                return true;
            }
        }
    return false;

}

MoveListe &MoveListe::operator=(const MoveListe &other) {
    for (auto i = 0; i < other.moveCounter; ++i) {
        liste[i] = other.liste[i];
    }
    this->moveCounter = other.moveCounter;
    return *this;
}



