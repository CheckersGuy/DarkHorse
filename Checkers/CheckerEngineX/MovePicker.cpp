//
// Created by Robin on 19.01.2018.
//

#include "MovePicker.h"
namespace Statistics {


    MovePicker mPicker;

    void MovePicker::clearScores() {
        std::fill(bfScore.begin(),bfScore.end(),0);
        std::fill(history.begin(),history.end(),0);
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
            return std::numeric_limits<int16_t>::max();
        }
        return getMoveScore(move, color);
    }

    void MovePicker::updateHHScore(Move move, Color color, int depth) {
        const int colorIndex = (color + 1) / 2;
        history[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()] += depth * depth;
    }

    void MovePicker::updateBFScore(Move* liste, int moveIndex, Color color, int depth) {
        const int colorIndex = (color + 1) / 2;
        for (auto i = 0; i < moveIndex; ++i) {
            bfScore[32 * 32 * colorIndex + 32 * liste[i].getToIndex() + liste[i].getFromIndex()] += depth;
        }
    }

    void MovePicker::update_scores(Move *list, int move_index, Color color, int depth) {
        Statistics::mPicker.updateHHScore(list[move_index], color, depth);
        Statistics::mPicker.updateBFScore(list, move_index, color, depth);
    }

}