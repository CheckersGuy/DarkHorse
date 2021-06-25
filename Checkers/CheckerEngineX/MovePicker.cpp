//
// Created by Robin on 19.01.2018.
//

#include "MovePicker.h"

namespace Statistics {


#ifndef TRAIN
    MovePicker mPicker;
#else
    thread_local MovePicker mPicker;
#endif


    void MovePicker::clearScores() {
        std::fill(bfScore.begin(), bfScore.end(), 0);
        std::fill(history.begin(), history.end(), 0);
    }

    int MovePicker::getMoveScore(Move move, Color color) {
        const int colorIndex = (color + 1) / 2;
        const int bScore = bfScore[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()] + 1;
        const int hhScore = history[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()];
        const int score = hhScore / bScore;

        return score;
    }

    int MovePicker::getMoveScore(Position current, Move move, Color color, Move ttMove) {

        if (move == ttMove) {
            return std::numeric_limits<int16_t>::max();
        }
        if (move.isCapture()) {
            uint32_t king_cap, pawn_cap;
            king_cap = Bits::pop_count(current.K & move.captures);
            pawn_cap = Bits::pop_count((current.WP | current.BP) & move.captures & (~current.K));

            return 10 * (int) pawn_cap + 12 * (int) king_cap;
        }

        return getMoveScore(move, color);
    }


    void MovePicker::update_scores(Move *liste, Move move, Color color, int depth) {
        const int colorIndex = (color + 1) / 2;
        history[32 * 32 * colorIndex + 32 * move.getToIndex() + move.getFromIndex()] += depth * depth;
        while (move != (*liste)) {
            Move top = *liste;
            bfScore[32 * 32 * colorIndex + 32 * top.getToIndex() + top.getFromIndex()] += depth;
            liste++;
        }

    }
}