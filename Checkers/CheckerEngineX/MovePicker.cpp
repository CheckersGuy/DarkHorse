//
// Created by Robin on 19.01.2018.
//

#include "MovePicker.h"

namespace Statistics {


    MovePicker mPicker;

    void MovePicker::clearScores() {
        for (int i = 0; i < 32 * 32 * 2; ++i) {
            this->bfScore[i] = 0;
            this->history[i] = 0;
        }
    }

    int MovePicker::getMoveScore(Move move, Color color) {
        const int colorIndex = (color + 1) / 2;
        const int bScore = bfScore[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()];
        const int hhScore = history[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()];
        const int score = (bScore == 0) ? 0 : ((hhScore / bScore));
        return score;
    }

    int MovePicker::getMoveScore(Move move, Color color, Move ttMove) {
        if (move == ttMove) {
            return 2000000;
        }
        int score=getMoveScore(move, color);

        return score;
    }

    void MovePicker::updateHHScore(Move move, Color color, int depth) {
        const int colorIndex = (color + 1) / 2;
        history[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()] += depth * depth;
    }

    void MovePicker::updateBFScore(Move *liste, int moveIndex, Color color, int depth) {
        const int colorIndex = (color + 1) / 2;
        for (auto i = 0; i < moveIndex; ++i) {
            bfScore[32 * 32 * colorIndex + 32 * liste[i].getToIndex() + liste[i].getFromIndex()] += depth;
        }
    }

}