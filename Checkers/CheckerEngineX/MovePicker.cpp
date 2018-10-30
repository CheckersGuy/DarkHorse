//
// Created by Robin on 19.01.2018.
//

#include "MovePicker.h"

namespace Statistics {

#ifdef TRAIN
    thread_local MovePicker mPicker;
#else
    MovePicker mPicker;
#endif


    void MovePicker::clearScores() {
        for (int i = 0; i < 32 * 32 * 2; ++i) {
            this->bfScore[i] = 0;
            this->history[i] = 0;
        }
    }

    int MovePicker::getMoveScore(Move move, Color color) {
        const int colorIndex = (color == BLACK) ? 0 : 1;
        const int bScore = bfScore[32 * 32 * colorIndex + 32 * move.getTo() + move.getFrom()];
        const int hhScore = history[32 * 32 * colorIndex + 32 * move.getTo() + move.getFrom()];
        const int score = (bScore == 0) ? -100 : ((10 * hhScore / bScore));
        return score;
    }

    int MovePicker::getMoveScore(Move move, Color color, Move ttMove) {
        int score = getMoveScore(move, color);
        if (move == ttMove && move.getMoveIndex() == ttMove.getMoveIndex()) {
            score += 200000;
        }

        return score;
    }


    int MovePicker::getHHScore(Move move, Color color) {
        const int colorIndex = (color == BLACK) ? 0 : 1;
        const int hhScore = history[32 * 32 * colorIndex + 32 * move.getTo() + move.getFrom()];
        return hhScore;
    }

    int MovePicker::getBFScore(Move move, Color color) {
        const int colorIndex = (color == BLACK) ? 0 : 1;
        const int bScore = bfScore[32 * 32 * colorIndex + 32 * move.getTo() + move.getFrom()];
        return bScore;
    }

    void MovePicker::updateHHScore(Move move, Color color, int depth) {
        const int colorIndex = (color == BLACK) ? 0 : 1;
        history[32 * 32 * colorIndex + 32 * move.getTo() + move.getFrom()] += depth * depth;
    }

    void MovePicker::updateBFScore(Move *liste, int moveIndex, Color color, int depth) {
        //needs to be implemented
        assert(moveIndex < 40);
        const int colorIndex = (color == BLACK) ? 0 : 1;
        for (int i = 0; i < moveIndex; ++i) {
            bfScore[32 * 32 * colorIndex + 32 * liste[i].getTo() + liste[i].getFrom()] += depth;
        }
    }

}